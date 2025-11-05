import SwiftUI

// --- 1. Data Structures to Match JSON Response ---
struct EngineHealthResponse: Codable {
    let overall_status: String
    let parts_health: [String: Double]
    let recommendations: [Recommendation]
    let current_readings: [String: Double]
}

struct Recommendation: Codable, Identifiable {
    let id = UUID()
    let part: String
    let current_health: String
    let priority: String
    let priority_level: String
    let recommended_replacement_date: String
    let remaining_kilometers: Double
    let remaining_km_formatted: String
    let approx_time_left: String

    private enum CodingKeys: String, CodingKey {
        case part, current_health, priority, priority_level, recommended_replacement_date, remaining_kilometers, remaining_km_formatted, approx_time_left
    }
}

// --- 2. SwiftUI View ---
struct ContentView: View {
    @State private var overallStatus: String = "Loading..."
    @State private var partsHealth: [String: Double] = [:]
    @State private var recommendations: [Recommendation] = []
    @State private var currentReadings: [String: Double] = [:]
    @State private var isLoading: Bool = true // Start in loading state
    @State private var errorMessage: String? = nil

    // IMPORTANT: Make sure this IP is correct!
    // This should be the IP address from your app_backend.py terminal (e.g., 192.168.68.102)
    let backendURL = "http://192.168.68.102:5000/predict"

    var body: some View {
        ZStack {
            // Main background color
            Color(red: 0.05, green: 0.07, blue: 0.1).ignoresSafeArea()
            
            // --- Logic to show Error, Loading, or Data ---
            
            if let error = errorMessage {
                // --- ERROR STATE ---
                VStack(spacing: 20) {
                    Image(systemName: "xmark.octagon.fill")
                        .font(.system(size: 60))
                        .foregroundColor(.red)
                    
                    Text("Error Occurred")
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                    
                    // Display the exact error message
                    ScrollView {
                        Text(error)
                            .font(.body)
                            .multilineTextAlignment(.center)
                            .foregroundColor(.gray)
                            .padding()
                    }
                    
                    Button("Retry Connection") {
                        fetchEngineHealth()
                    }
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .padding()
            }
            else if isLoading {
                // --- LOADING STATE ---
                VStack(spacing: 20) {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        .scaleEffect(2.0)
                    Text("Fetching Health Report...")
                        .font(.title3)
                        .foregroundColor(.gray)
                }
            }
            else {
                // --- SUCCESS STATE (Show Data) ---
                NavigationView {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 20) {

                            Text("Overall Engine Status")
                                .font(.title2).bold()
                            Text(overallStatus)
                                .font(.title)
                                .foregroundColor(statusColor(overallStatus))
                                .padding(.bottom)

                            Text("Individual Part Health")
                                .font(.title2).bold()
                            
                            ForEach(partsHealth.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                                HStack {
                                    Text("\(formatPartName(key)):")
                                        .frame(width: 150, alignment: .leading)
                                    Text("\(value, specifier: "%.1f")%")
                                        .foregroundColor(healthColor(value))
                                }
                            }
                            .padding(.bottom)

                            Text("Maintenance Recommendations")
                                .font(.title2).bold()
                            
                            ForEach(recommendations) { rec in // Already sorted by backend
                                VStack(alignment: .leading) {
                                    Text(rec.part).bold()
                                        .foregroundColor(healthColor(partsHealth[rec.part.lowercased().replacingOccurrences(of: " ", with: "_")]))
                                    Text("Health: \(rec.current_health)")
                                    Text("Status: \(rec.priority)")
                                    Text("Recommended Date: \(rec.recommended_replacement_date)")
                                    Text("Remaining Distance: \(rec.remaining_km_formatted)")
                                    Text("Approx. Time Left: \(rec.approx_time_left)")
                                }
                                .padding(.vertical, 5)
                                Divider().background(Color.gray)
                            }
                            
                            Spacer()
                        }
                        .padding()
                    }
                    .navigationTitle("Engine Health")
                    .toolbar {
                        ToolbarItem(placement: .navigationBarTrailing) {
                            Button {
                                fetchEngineHealth()
                            } label: {
                                Image(systemName: "arrow.clockwise")
                            }
                            .disabled(isLoading)
                        }
                    }
                    .background(Color.clear)
                    .foregroundColor(.white)
                }
                .navigationViewStyle(.stack)
                .toolbarColorScheme(.dark, for: .navigationBar)
            }
        }
        .onAppear {
            // Fetch data when the view first appears
            fetchEngineHealth()
        }
    }

    // --- 3. Function to Fetch Data from Backend ---
    func fetchEngineHealth() {
        print("Fetching engine health...")
        isLoading = true
        errorMessage = nil

        guard let url = URL(string: backendURL) else {
            errorMessage = "Invalid backend URL: \(backendURL)"
            isLoading = false
            print(errorMessage!)
            return
        }

        let sampleSensorData: [String: Double] = [
            "engineRpm": Double.random(in: 500...3000),
            "lubOilPressure": Double.random(in: 1.0...70.0),
            "fuelPressure": Double.random(in: 1.0...8.0),
            "coolantPressure": Double.random(in: 0.5...4.0),
            "lubOilTemp": Double.random(in: 60.0...110.0),
            "coolantTemp": Double.random(in: 70.0...105.0)
        ]

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 10

        do {
            request.httpBody = try JSONEncoder().encode(sampleSensorData)
            print("Sending data to backend: \(String(data: request.httpBody!, encoding: .utf8) ?? "Invalid JSON")")
        } catch {
            errorMessage = "Failed to encode sensor data: \(error.localizedDescription)"
            isLoading = false
            print(errorMessage!)
            return
        }

        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                
                if let error = error {
                    errorMessage = "Network request failed: \(error.localizedDescription)"
                    isLoading = false
                    print(errorMessage!)
                    if let urlError = error as? URLError {
                         print("URLError Code: \(urlError.errorCode)")
                    }
                    return
                }

                guard let httpResponse = response as? HTTPURLResponse else {
                    errorMessage = "Invalid response from server (Not HTTP)"
                    isLoading = false
                    print(errorMessage!)
                    return
                }

                guard httpResponse.statusCode == 200 else {
                    errorMessage = "Server returned error (Status code: \(httpResponse.statusCode))"
                    isLoading = false
                     print(errorMessage!)
                    if let responseData = data, let responseString = String(data: responseData, encoding: .utf8) {
                        print("Server error response data: \(responseString)")
                    }
                    return
                }

                guard let data = data else {
                    errorMessage = "No data received from server"
                    isLoading = false
                    print(errorMessage!)
                    return
                }
                 print("Received data from backend: \(String(data: data, encoding: .utf8) ?? "Invalid data")")

                do {
                    let decoder = JSONDecoder()
                    let healthResponse = try decoder.decode(EngineHealthResponse.self, from: data)

                    self.overallStatus = healthResponse.overall_status
                    self.partsHealth = healthResponse.parts_health
                    self.recommendations = healthResponse.recommendations
                    self.currentReadings = healthResponse.current_readings
                    self.errorMessage = nil
                    self.isLoading = false // <<<<--- STOP LOADING ON SUCCESS
                    print("Successfully decoded response and updated UI state.")

                } catch let decodingError as DecodingError {
                     switch decodingError {
                     case .typeMismatch(let type, let context):
                          errorMessage = "JSON Decode Error: Type mismatch for key '\(context.codingPath.last?.stringValue ?? "N/A")'. Expected \(type)."
                     case .valueNotFound(let type, let context):
                          errorMessage = "JSON Decode Error: Value not found for key '\(context.codingPath.last?.stringValue ?? "N/A")'. Expected \(type)."
                     case .keyNotFound(let key, let context):
                          errorMessage = "JSON Decode Error: Key '\(key.stringValue)' not found."
                     case .dataCorrupted(let context):
                          errorMessage = "JSON Decode Error: Data corrupted."
                     @unknown default:
                          errorMessage = "JSON Decode Error: Unknown error."
                     }
                    self.isLoading = false // <<<<--- STOP LOADING ON ERROR
                    print("Decoding Error: \(errorMessage!)")
                    print("Raw Data causing error: \(String(data: data, encoding: .utf8) ?? "Unable to decode data")")
                } catch {
                     errorMessage = "Failed to decode JSON response: \(error.localizedDescription)"
                     self.isLoading = false // <<<<--- STOP LOADING ON ERROR
                     print("\(errorMessage!). Raw Data: \(String(data: data, encoding: .utf8) ?? "Unable to decode data")")
                }
            }
        }.resume()
    }

    // --- Helper Functions for UI ---
    func statusColor(_ status: String) -> Color {
        switch status.lowercased() {
        case "good": return .green
        case "moderate": return .orange
        case "bad": return .red
        default: return .gray
        }
    }
    
    func healthColor(_ score: Double?) -> Color {
        guard let score = score else { return .gray }
        if score > 70 { return .green }
        if score > 40 { return .orange }
        return .red
    }

    func formatPartName(_ key: String) -> String {
        return key.replacingOccurrences(of: "_", with: " ").capitalized
    }
}

// --- Preview Provider (for Xcode Canvas) ---
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .preferredColorScheme(.dark)
    }
}
