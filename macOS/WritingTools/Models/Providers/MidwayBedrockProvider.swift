import Foundation
import Observation

private let logger = AppLogger.logger("MidwayBedrockProvider")

// MARK: - Config

struct MidwayBedrockConfig: Sendable {
    var endpointURL: String
    var modelId: String

    /// Dex AIDP Fargate inference endpoint (Bedrock InvokeModel envelope, Midway-authed).
    static let defaultEndpoint = "https://prod.fargate.inference.aidp.dex.amazon.dev/"
    static let defaultModelId = "global.anthropic.claude-sonnet-4-6"
}

// Model IDs verified live against the Dex Fargate endpoint. The Global CRIS IDs
// use a bare form for 4.6 (no date, no `-v1:0`); the `us.` prefix and `-v1:0`
// suffix variants 500 there.
enum MidwayBedrockModel: String, CaseIterable {
    case sonnet46 = "global.anthropic.claude-sonnet-4-6"
    case sonnet45 = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
    case haiku45 = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    case custom

    var displayName: String {
        switch self {
        case .sonnet46: return "Claude Sonnet 4.6 (Bedrock, Global CRIS)"
        case .sonnet45: return "Claude Sonnet 4.5 (Bedrock, 1M Context)"
        case .haiku45: return "Claude Haiku 4.5 (Bedrock, Faster)"
        case .custom: return "Custom"
        }
    }
}

// MARK: - Provider

@Observable
@MainActor
final class MidwayBedrockProvider: AIProvider {
    var isProcessing: Bool = false

    private let config: MidwayBedrockConfig
    private var activeTask: Task<Void, any Error>?

    init(config: MidwayBedrockConfig) {
        self.config = config
    }

    func processText(
        systemPrompt: String?,
        userPrompt: String,
        images: [Data],
        streaming: Bool
    ) async throws -> String {
        isProcessing = true
        defer { isProcessing = false }

        var compiled = ""
        let request = try await Self.buildRequest(
            config: config,
            systemPrompt: systemPrompt,
            userPrompt: userPrompt,
            images: images
        )
        try await Self.streamResponse(for: request, forceRefreshOn401: true) { text in
            compiled += text
        }

        guard !compiled.isEmpty else {
            throw MidwayBedrockError.invalidResponse("No text content in response.")
        }
        return compiled
    }

    func processTextStreaming(
        systemPrompt: String?,
        userPrompt: String,
        images: [Data],
        onChunk: @escaping @Sendable @MainActor (String) -> Void
    ) async throws {
        isProcessing = true
        defer {
            isProcessing = false
            activeTask = nil
        }

        let config = self.config
        let streamTask = Task { @MainActor in
            let request = try await Self.buildRequest(
                config: config,
                systemPrompt: systemPrompt,
                userPrompt: userPrompt,
                images: images
            )
            try await Self.streamResponse(for: request, forceRefreshOn401: true, onText: onChunk)
        }
        activeTask = streamTask
        try await streamTask.value
    }

    func cancel() {
        activeTask?.cancel()
        activeTask = nil
        isProcessing = false
    }

    // MARK: - Request building

    nonisolated private static func buildRequest(
        config: MidwayBedrockConfig,
        systemPrompt: String?,
        userPrompt: String,
        images: [Data]
    ) async throws -> URLRequest {
        guard !config.endpointURL.isEmpty, let url = URL(string: config.endpointURL) else {
            throw MidwayBedrockError.invalidConfiguration("Invalid endpoint URL.")
        }
        guard !config.modelId.isEmpty else {
            throw MidwayBedrockError.invalidConfiguration("Model ID is required.")
        }

        let token = try await MidwayTokenStore.shared.token()

        // Anthropic-on-Bedrock content blocks: text first, then any images.
        var contentBlocks: [[String: Any]] = [
            ["type": "text", "text": userPrompt]
        ]
        for imageData in images {
            contentBlocks.append([
                "type": "image",
                "source": [
                    "type": "base64",
                    "media_type": detectImageMIMEType(imageData),
                    "data": imageData.base64EncodedString()
                ]
            ])
        }

        var body: [String: Any] = [
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 10000,
            "temperature": 0.7,
            "messages": [
                ["role": "user", "content": contentBlocks]
            ]
        ]
        if let systemPrompt, !systemPrompt.isEmpty {
            body["system"] = systemPrompt
        }

        let envelope: [String: Any] = [
            "modelId": config.modelId,
            "body": body
        ]

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 120
        request.httpBody = try JSONSerialization.data(withJSONObject: envelope)
        return request
    }

    // MARK: - Streaming

    /// Streams the Bedrock InvokeModelWithResponseStream response. The body is a
    /// run of concatenated top-level JSON objects (NOT SSE), e.g.
    /// `{message_start}{content_block_delta}...{message_stop}`. We split them by
    /// brace depth and forward `text_delta` chunks.
    @MainActor
    private static func streamResponse(
        for request: URLRequest,
        forceRefreshOn401: Bool,
        onText: @escaping @MainActor (String) -> Void
    ) async throws {
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw MidwayBedrockError.networkError("Invalid response from server.")
        }

        if http.statusCode == 401 && forceRefreshOn401 {
            // Token may have expired mid-flight; mint a fresh one and retry once.
            await MidwayTokenStore.shared.invalidate()
            let token = try await MidwayTokenStore.shared.token()
            var retry = request
            retry.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            try await streamResponse(for: retry, forceRefreshOn401: false, onText: onText)
            return
        }

        guard (200...299).contains(http.statusCode) else {
            var data = Data()
            for try await byte in bytes { data.append(byte) }
            let detail = String(data: data, encoding: .utf8) ?? "HTTP \(http.statusCode)"
            throw MidwayBedrockError.apiError("API Error (\(http.statusCode)): \(detail)")
        }

        // An expired Midway session can yield a 200 HTML login page instead of the
        // event stream. Detect it the same way the reference web client does.
        if let contentType = http.value(forHTTPHeaderField: "Content-Type"),
           contentType.localizedCaseInsensitiveContains("text/html") {
            await MidwayTokenStore.shared.invalidate()
            throw MidwayBedrockError.authFailed(
                "Received a login page instead of an API response. Run `mwinit` in Terminal and retry."
            )
        }

        var scanner = BedrockJSONStreamScanner()
        for try await byte in bytes {
            try Task.checkCancellation()
            guard let object = scanner.push(byte) else { continue }
            guard let json = try? JSONSerialization.jsonObject(with: object) as? [String: Any] else {
                continue
            }
            // A top-level `error` field signals a stream error (matches the
            // reference processEvent, which keys on event.error regardless of type).
            if let error = json["error"] {
                let message = (error as? [String: Any])?["message"] as? String
                    ?? (error as? String)
                    ?? "Unknown streaming error."
                throw MidwayBedrockError.apiError(message)
            }
            if json["type"] as? String == "content_block_delta",
               let delta = json["delta"] as? [String: Any],
               delta["type"] as? String == "text_delta",
               let text = delta["text"] as? String {
                onText(text)
            }
            // message_start / content_block_start / content_block_stop /
            // message_delta / message_stop carry no user-visible text — ignore.
        }
    }
}

// MARK: - Streaming JSON object scanner

/// Incrementally slices a byte stream of concatenated top-level JSON objects
/// into individual object payloads, tracking brace depth while respecting
/// string literals and escapes. The four delimiter bytes (`{` `}` `"` `\`) are
/// all ASCII (< 0x80), so byte-level scanning is UTF-8 safe.
private struct BedrockJSONStreamScanner {
    private var buffer: [UInt8] = []
    private var depth = 0
    private var inString = false
    private var escaped = false
    private var objectStart: Int?

    /// Feeds one byte. Returns a complete top-level object's bytes when the
    /// closing brace lands, otherwise nil.
    mutating func push(_ byte: UInt8) -> Data? {
        let index = buffer.count
        buffer.append(byte)

        if inString {
            if escaped {
                escaped = false
            } else if byte == 0x5C { // backslash
                escaped = true
            } else if byte == 0x22 { // quote
                inString = false
            }
            return nil
        }

        switch byte {
        case 0x22: // quote
            inString = true
        case 0x7B: // {
            if depth == 0 { objectStart = index }
            depth += 1
        case 0x7D: // }
            if depth > 0 { depth -= 1 }
            if depth == 0, let start = objectStart {
                let slice = Data(buffer[start...index])
                objectStart = nil
                buffer.removeFirst(index + 1)
                return slice
            }
        default:
            break
        }
        return nil
    }
}

// MARK: - Token store

/// Mints and caches a Midway OIDC `id_token` by replaying the same flow as the
/// shell `mcurl .../SSO` command: it shells out to `/usr/bin/curl` using the
/// Midway cookie jar at `~/.midway/cookie`. Requires the app sandbox to be off.
actor MidwayTokenStore {
    static let shared = MidwayTokenStore()

    private var cachedToken: String?
    private var expiresAt: Date?

    /// Refresh a little before the real expiry to avoid mid-flight 401s.
    private static let expiryGuard: TimeInterval = 300
    private static let fallbackLifetime: TimeInterval = 50 * 60

    func token() async throws -> String {
        if let token = cachedToken,
           let expiresAt,
           expiresAt.timeIntervalSinceNow > Self.expiryGuard {
            return token
        }
        let token = try await Self.mint()
        cachedToken = token
        expiresAt = Self.expiry(of: token) ?? Date().addingTimeInterval(Self.fallbackLifetime)
        return token
    }

    func invalidate() {
        cachedToken = nil
        expiresAt = nil
    }

    // MARK: Minting

    private static var cookiePath: String {
        ("~/.midway/cookie" as NSString).expandingTildeInPath
    }

    private static func mint() async throws -> String {
        let cookie = cookiePath
        guard FileManager.default.fileExists(atPath: cookie) else {
            throw MidwayBedrockError.authFailed(
                "Midway cookie not found at \(cookie). Run `mwinit` in Terminal first."
            )
        }

        let nonce = UUID().uuidString
        let output = try await runCurl(arguments: [
            "-sL",
            "--cookie", cookie,
            "--cookie-jar", cookie,
            "-G", "https://midway-auth.amazon.com/SSO",
            "--data-urlencode", "response_type=id_token",
            "--data-urlencode", "scope=openid",
            "--data-urlencode", "client_id=https://ai-hub.dex.amazon.dev",
            "--data-urlencode", "redirect_uri=https://ai-hub.dex.amazon.dev/",
            "--data-urlencode", "nonce=\(nonce)"
        ])

        let token = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard isLikelyJWT(token) else {
            // Expired cookie typically returns an HTML login page instead of a token.
            throw MidwayBedrockError.authFailed(
                "Midway returned no valid token (cookie likely expired). Run `mwinit` in Terminal and retry."
            )
        }
        logger.debug("Minted Midway id_token (length: \(token.count)).")
        return token
    }

    private static func isLikelyJWT(_ token: String) -> Bool {
        guard token.count > 100, !token.contains(" "), !token.contains("\n") else { return false }
        return token.filter { $0 == "." }.count == 2
    }

    /// Decodes the JWT `exp` claim (unix seconds) into an absolute date.
    private static func expiry(of token: String) -> Date? {
        let segments = token.split(separator: ".")
        guard segments.count == 3 else { return nil }
        var base64 = String(segments[1])
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        while base64.count % 4 != 0 { base64 += "=" }
        guard let data = Data(base64Encoded: base64),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let exp = json["exp"] as? Double else {
            return nil
        }
        return Date(timeIntervalSince1970: exp)
    }

    /// Runs `/usr/bin/curl` off the main thread and returns its stdout.
    private static func runCurl(arguments: [String]) async throws -> String {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                process.executableURL = URL(fileURLWithPath: "/usr/bin/curl")
                process.arguments = arguments

                let stdout = Pipe()
                let stderr = Pipe()
                process.standardOutput = stdout
                process.standardError = stderr

                do {
                    try process.run()
                } catch {
                    continuation.resume(throwing: MidwayBedrockError.authFailed(
                        "Failed to launch curl: \(error.localizedDescription)"
                    ))
                    return
                }

                let outData = stdout.fileHandleForReading.readDataToEndOfFile()
                let errData = stderr.fileHandleForReading.readDataToEndOfFile()
                process.waitUntilExit()

                guard process.terminationStatus == 0 else {
                    let detail = String(data: errData, encoding: .utf8)?
                        .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                    continuation.resume(throwing: MidwayBedrockError.authFailed(
                        "curl exited with status \(process.terminationStatus). \(detail)"
                    ))
                    return
                }
                continuation.resume(returning: String(data: outData, encoding: .utf8) ?? "")
            }
        }
    }
}

// MARK: - Errors

enum MidwayBedrockError: LocalizedError {
    case invalidConfiguration(String)
    case authFailed(String)
    case networkError(String)
    case apiError(String)
    case invalidResponse(String)

    var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message): return "Configuration Error: \(message)"
        case .authFailed(let message): return "Midway Auth Error: \(message)"
        case .networkError(let message): return "Network Error: \(message)"
        case .apiError(let message): return message
        case .invalidResponse(let message): return "Response Error: \(message)"
        }
    }
}
