//
//  MidwaySettingsView.swift
//  WritingTools
//

import SwiftUI
import AppKit

struct MidwaySettingsView: View {
    @Bindable var settings = AppSettings.shared
    @Binding var needsSaving: Bool
    @State private var modelSelection: MidwayBedrockModel

    init(needsSaving: Binding<Bool>) {
        self._needsSaving = needsSaving
        let current = AppSettings.shared.midwayModelId
        if let known = MidwayBedrockModel(rawValue: current), known != .custom {
            self._modelSelection = State(initialValue: known)
        } else {
            self._modelSelection = State(initialValue: .custom)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Endpoint")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                TextField("Inference Endpoint URL", text: $settings.midwayEndpointURL)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
                    .onChange(of: settings.midwayEndpointURL) { _, _ in needsSaving = true }

                Text("Authentication is automatic via your local Midway cookie (~/.midway/cookie). If requests fail with an auth error, run `mwinit` in Terminal.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Model Selection")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                Picker("Model", selection: $modelSelection) {
                    ForEach(MidwayBedrockModel.allCases, id: \.self) { model in
                        Text(model.displayName).tag(model)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity, alignment: .leading)
                .onChange(of: modelSelection) { _, newValue in
                    if newValue != .custom {
                        settings.midwayModelId = newValue.rawValue
                    }
                    needsSaving = true
                }

                if modelSelection == .custom {
                    TextField("Custom Bedrock Model ID", text: $settings.midwayModelId)
                        .textFieldStyle(.roundedBorder)
                        .font(.caption)
                        .onChange(of: settings.midwayModelId) { _, _ in needsSaving = true }
                }

                Text("E.g., \(MidwayBedrockModel.sonnet46.rawValue)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .onAppear { syncModelSelection(settings.midwayModelId) }
        .onChange(of: settings.midwayModelId) { _, newValue in syncModelSelection(newValue) }
    }

    private func syncModelSelection(_ modelId: String) {
        if let known = MidwayBedrockModel(rawValue: modelId), known != .custom {
            modelSelection = known
        } else {
            modelSelection = .custom
        }
    }
}
