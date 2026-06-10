//
//  MidwayProviderSettingsView.swift
//  WritingTools
//

import SwiftUI

struct MidwayProviderSettingsView: View {
    @Bindable var settings: AppSettings
    @State private var modelSelection: MidwayBedrockModel = .sonnet46

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Configure Midway (Bedrock)")
                .font(.headline)

            TextField("Inference Endpoint URL", text: $settings.midwayEndpointURL)
                .textFieldStyle(.roundedBorder)
                .font(.caption)

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
            }

            if modelSelection == .custom {
                TextField("Custom Bedrock Model ID", text: $settings.midwayModelId)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
            }

            Text("Authentication is automatic via your local Midway cookie. Run `mwinit` in Terminal if requests fail with an auth error.")
                .font(.caption)
                .foregroundStyle(.secondary)
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
