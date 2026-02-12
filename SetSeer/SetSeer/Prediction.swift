//
//  Prediction.swift
//  SetSeer
//
//  Created by Duncan on 8/15/25.
//

import Foundation
import CoreGraphics

// A structure to hold the results for a single detected card.
// `Identifiable` is useful for SwiftUI's ForEach.
struct Prediction: Identifiable {
    let id = UUID()
    let boundingBox: CGRect
    let label: String
    let confidence: Float
}

// MARK: - Enums for Card Features

enum SetNumber: Int, CaseIterable, Equatable {
    case one = 1
    case two = 2
    case three = 3
}

enum SetColor: String, CaseIterable, Equatable {
    case red
    case green
    case purple
}

enum SetShape: String, CaseIterable, Equatable {
    case diamond
    case oval
    case squiggle
}

enum SetShading: String, CaseIterable, Equatable {
    case solid
    case striped
    case empty
}

// MARK: - Card Model

struct SetCard: Identifiable, Equatable, Hashable {
    let id: UUID
    let number: SetNumber
    let color: SetColor
    let shape: SetShape
    let shading: SetShading
    let boundingBox: CGRect // Normalized [0,1]
    
    static func == (lhs: SetCard, rhs: SetCard) -> Bool {
        return lhs.number == rhs.number &&
               lhs.color == rhs.color &&
               lhs.shape == rhs.shape &&
               lhs.shading == rhs.shading &&
               lhs.id == rhs.id
    }
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

struct SetResult: Identifiable {
    let id = UUID()
    let cards: [SetCard]
    var colorIndex: Int = 0 
}

// MARK: - Game Engine

class SetGameEngine {
    
    static func parse(prediction: Prediction) -> SetCard? {
        let rawParts = prediction.label.lowercased().replacingOccurrences(of: "-", with: " ").split(separator: " ").map { String($0) }
        
        guard rawParts.count >= 4 else { return nil }
        
        guard let numberVal = Int(rawParts[0]), let number = SetNumber(rawValue: numberVal) else { return nil }
        guard let color = SetColor(rawValue: rawParts[1]) else { return nil }
        guard let shading = SetShading(rawValue: rawParts[2]) else { return nil }
        guard let shape = SetShape(rawValue: rawParts[3]) else { return nil }
        
        return SetCard(
            id: prediction.id,
            number: number,
            color: color,
            shape: shape,
            shading: shading,
            boundingBox: prediction.boundingBox
        )
    }
    
    static func findSets(cards: [SetCard]) -> [SetResult] {
        var results: [SetResult] = []
        let n = cards.count
        
        if n < 3 { return [] }
        
        var cardMap: [String: [SetCard]] = [:]
        
        for card in cards {
            let key = signature(for: card)
            cardMap[key, default: []].append(card)
        }
        
        var foundSetIds = Set<String>() 
        
        for i in 0..<n {
            for j in (i+1)..<n {
                let cardA = cards[i]
                let cardB = cards[j]
                
                let targetNumber = getCompletingFeature(cardA.number, cardB.number)
                let targetColor = getCompletingFeature(cardA.color, cardB.color)
                let targetShading = getCompletingFeature(cardA.shading, cardB.shading)
                let targetShape = getCompletingFeature(cardA.shape, cardB.shape)
                
                let targetSig = "\(targetNumber.rawValue)-\(targetColor.rawValue)-\(targetShading.rawValue)-\(targetShape.rawValue)"
                
                if let potentialMatches = cardMap[targetSig] {
                    for cardC in potentialMatches {
                        if cardC.id != cardA.id && cardC.id != cardB.id {
                            let setIds = [cardA.id, cardB.id, cardC.id].map { $0.uuidString }.sorted()
                            let setSignature = setIds.joined(separator: ",")
                            
                            if !foundSetIds.contains(setSignature) {
                                foundSetIds.insert(setSignature)
                                results.append(SetResult(cards: [cardA, cardB, cardC]))
                            }
                        }
                    }
                }
            }
        }
        
        return results
    }
    
    private static func signature(for card: SetCard) -> String {
        return "\(card.number.rawValue)-\(card.color.rawValue)-\(card.shading.rawValue)-\(card.shape.rawValue)"
    }
    
    private static func getCompletingFeature<T: Equatable & CaseIterable>(_ a: T, _ b: T) -> T {
        if a == b { return a }
        for caseVal in T.allCases {
            if caseVal != a && caseVal != b {
                return caseVal
            }
        }
        return a 
    }
}
