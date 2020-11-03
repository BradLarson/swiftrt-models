// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
import ModelSupport
import SwiftRT

//let stepCount = 5
let stepCount = 1
//let gridSize = 512
let gridSize = 10
//let particleCount = 1024
let particleCount = 5
let senseAngle = 0.20 * Float.pi
let senseDistance: Float = 4.0
let evaporationRate: Float = 0.95
let moveAngle = 0.1 * Float.pi
let moveStep: Float = 2.0
let channelSize = 1
let captureImage = true
let runOnCPU = false

if runOnCPU { use(device: 0) }

var grid = [repeating(0, (gridSize, gridSize), type: Float.self),
            repeating(0, (gridSize, gridSize), type: Float.self)]
var positions = TensorR2<Float>(randomUniform: Shape2(particleCount, 2)) * Float(gridSize)
var headings = TensorR1<Float>(randomUniform: particleCount) * 2.0 * Float.pi

let gridShape = repeating(array([Int32(gridSize), Int32(gridSize)], (1, 1, 2)), (particleCount, 3, 2))

extension Tensor where TensorElement.Value: Numeric {
  func mask(condition: (Tensor<Shape, TensorElement>) -> Tensor<Shape, Bool>) -> Tensor<Shape, TensorElement> {
    let satisfied = condition(self)
    return replace(x: zeros(like: self), with: ones(like: self), where: satisfied)
  }
}

extension TensorR2 where TensorElement.Value: Real {
  func angleToVector() -> TensorR3<TensorElement> {
    // Note: Axis of -1 produced wrong shape here.
    return TensorR3(stacking: [cos(self), sin(self)], axis: 2)
  }
}

func step(phase: Int) {
  var currentGrid = grid[phase]

  // Perceive
  let senseDirection = repeating(expand(dims: headings, axis: 1), (particleCount, 3)) +
    repeating(array([-moveAngle, 0.0, moveAngle], (1, 3)), (particleCount, 3))

  let sensingOffset = senseDirection.angleToVector() * senseDistance
  let sensingPosition = repeating(expand(dims: positions, axis: 1), (particleCount, 3, 2)) + sensingOffset
  // TODO: This wrapping around negative values needs to be fixed.
  let sensingIndices = abs(TensorR3<Int32>(sensingPosition))
  // TODO: Add the following modulus to the above.
  //  % gridShape

  // TODO: Gather
  //  let sensedValues = currentGrid.expandingShape(at: 2)
  //    .dimensionGathering(atIndices: sensingIndices).squeezingShape(at: 2)

  let sensedValues = array([-3.0, 0.0, 2.0,
                            1.0, 1.0, 1.0,
                            1.0, 2.0, 3.0,
                            3.0, 2.0, 1.0,
                            -1.0, -2.0, -3.0], (5, 3))

  // Move

  // lowValues should be [0, 0-2, 0, 2, 2]
  // highValues should be [2, 0-2, 2, 0, 0]
  let lowValues = argmin(sensedValues)
  let highValues = argmax(sensedValues)
  
  print("lowValues shape: \(lowValues.shape)")
  print("lowValues: \(lowValues)")
//  let middleMask = lowValues.mask { $0 .== 1 }
  
  let middleDistribution = TensorR1<Float>(randomUniform: particleCount)
  let randomTurn = middleDistribution.mask { $0 .< 0.1 }
    // * TensorR1<Float>(middleMask)
//  let turn = TensorR1<Float>(highValues - 1) * TensorR1<Float>(1 - middleMask) + randomTurn
//  headings += (turn * moveAngle)
//  positions += angleToVector(headings) * moveStep

  /*
  
  // Deposit
  // TODO: This wrapping around negative values needs to be fixed.
  let depositIndices = abs(Tensor<Int32>(positions)) % (gridShape.expandingShape(at: 0))
  let deposits = scatterValues.dimensionScattering(atIndices: depositIndices, shape: gridShape)
  currentGrid += deposits
  
  // Diffuse
  currentGrid = currentGrid.expandingShape(at: 0).expandingShape(at: 3)
  currentGrid = currentGrid.padded(forSizes: [(0, 0), (1, 1), (1, 1), (0, 0)], mode: .reflect)
  currentGrid = avgPool2D(currentGrid, filterSize: (1, 3, 3, 1), strides: (1, 1, 1, 1), padding: .valid)
  currentGrid = currentGrid * evaporationRate
  grid[1 - phase] = currentGrid.squeezingShape(at: 3).squeezingShape(at: 0)
  */
}

let start = Date()

var steps: [TensorR3<Float>] = []
for stepIndex in 0..<stepCount {
  step(phase: stepIndex % 2)
  if captureImage {
    steps.append(expand(dims: grid[0], axis: 2) * 255.0)
  }
}

print("Total calculation time: \(String(format: "%.4f", Date().timeIntervalSince(start))) seconds")

if captureImage {
  try steps.saveAnimatedImage(directory: "output", name: "physarum", delay: 1)
}

// TODO: argmin and argmax should take in comparable values and provide indices out.
// TODO: argmin and argmax shoulf provide a squeezing axis parameter.
@inlinable public func argmin<S,E>(
    _ lhs: Tensor<S,E>
) -> Tensor<S,E> where E.Value: Comparable & ComparableLimits {
  var result = Tensor(like: lhs)
  currentQueue.argmin(lhs, &result)
  return result
}

@inlinable public func argmax<S,E>(
    _ lhs: Tensor<S,E>
) -> Tensor<S,E> where E.Value: Comparable & ComparableLimits {
  var result = Tensor(like: lhs)
  currentQueue.argmax(lhs, &result)
  return result
}
