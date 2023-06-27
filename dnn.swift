import Foundation

// Aktivierungsfunktion: Sigmoid
func sigmoid(_ x: Double) -> Double {
    return 1 / (1 + exp(-x))
}

// Ableitung der Sigmoid-Funktion
func sigmoidDerivative(_ x: Double) -> Double {
    let sig = sigmoid(x)
    return sig * (1 - sig)
}

// Klasse für ein einfaches neuronales Netz
class NeuralNetwork {
    var inputNodes: Int
    var hiddenNodes: Int
    var outputNodes: Int
    
    var weightsIH: [[Double]]
    var weightsHO: [[Double]]
    var biasH: [Double]
    var biasO: [Double]
    
    init(inputNodes: Int, hiddenNodes: Int, outputNodes: Int) {
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        
        // Initialisiere Gewichte zwischen Eingabe- und versteckten Schichten zufällig
        self.weightsIH = Array(repeating: Array(repeating: 0, count: hiddenNodes), count: inputNodes)
        for i in 0..<inputNodes {
            for j in 0..<hiddenNodes {
                self.weightsIH[i][j] = Double.random(in: -1...1)
            }
        }
        
        // Initialisiere Gewichte zwischen versteckten und Ausgabeschichten zufällig
        self.weightsHO = Array(repeating: Array(repeating: 0, count: outputNodes), count: hiddenNodes)
        for i in 0..<hiddenNodes {
            for j in 0..<outputNodes {
                self.weightsHO[i][j] = Double.random(in: -1...1)
            }
        }
        
        // Initialisiere Bias-Werte zufällig
        self.biasH = Array(repeating: 0, count: hiddenNodes)
        self.biasO = Array(repeating: 0, count: outputNodes)
        for i in 0..<hiddenNodes {
            self.biasH[i] = Double.random(in: -1...1)
        }
        for i in 0..<outputNodes {
            self.biasO[i] = Double.random(in: -1...1)
        }
    }
    
    // Feedforward-Funktion
    func feedforward(_ inputArray: [Double]) -> [Double] {
        // Berechne Aktivierungen für versteckte Schicht
        var hiddenInputs: [Double] = Array(repeating: 0, count: hiddenNodes)
        for i in 0..<hiddenNodes {
            var sum = 0.0
            for j in 0..<inputNodes {
                sum += inputArray[j] * weightsIH[j][i]
            }
            sum += biasH[i]
            hiddenInputs[i] = sum
        }
        
        // Wende Sigmoid-Funktion auf versteckte Schicht an
        var hiddenOutputs: [Double] = Array(repeating: 0, count: hiddenNodes)
        for i in 0..<hiddenNodes {
            hiddenOutputs[i] = sigmoid(hiddenInputs[i])
        }
        
        // Berechne Aktivierungen für Ausgabeschicht
        var finalInputs: [Double] = Array(repeating: 0, count: outputNodes)
        for i in 0..<outputNodes {
            var sum = 0.0
            for j in 0..<hiddenNodes {
                sum += hiddenOutputs[j] * weightsHO[j][i]
            }
        }
    }
}
