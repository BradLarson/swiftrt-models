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

//let stepCount = 20
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

let gridShapeR3 = repeating(array([Int32(gridSize), Int32(gridSize)], (1, 1, 2)), (particleCount, 3, 2))
let gridShapeR2 = repeating(array([Int32(gridSize), Int32(gridSize)], (1, 2)), (particleCount, 2))

extension Tensor where TensorElement.Value: Numeric {
  func mask(condition: (Tensor<Shape, TensorElement>) -> Tensor<Shape, Bool>) -> Tensor<Shape, TensorElement> {
    return cast(condition(self), elementsTo: TensorElement.self)
  }
}

extension TensorR2 where TensorElement.Value: Real {
  func angleToVector() -> TensorR3<TensorElement> {
    return TensorR3(stacking: [cos(self), sin(self)], axis: -1)
  }
}

extension TensorR1 where TensorElement.Value: Real {
  func angleToVector() -> TensorR2<TensorElement> {
    return TensorR2(stacking: [cos(self), sin(self)], axis: -1)
  }
}

func step(phase: Int) {
  var currentGrid = grid[phase]

  // Perceive
  let senseDirection = repeating(expand(dims: headings, axis: 1), (particleCount, 3)) +
    repeating(array([-moveAngle, 0.0, moveAngle], (1, 3)), (particleCount, 3))
  // TODO: I shouldn't need to specify the tensor type here to make this unambiguous.
  let sensingOffset: TensorR3<Float> = senseDirection.angleToVector() * senseDistance
  let sensingPosition = repeating(expand(dims: positions, axis: 1), (particleCount, 3, 2)) + sensingOffset
  // TODO: This wrapping around negative values needs to be fixed.
  let sensingIndices = abs(cast(sensingPosition, elementsTo: Int32.self)) % gridShapeR3
  let sensedValues = gather(from: currentGrid, indices: sensingIndices)

  // Move
  let lowValues = argmin3(sensedValues)
  let highValues = argmax3(sensedValues)
  let middleMask = lowValues.mask { $0 .== 1 }
  let middleDistribution = TensorR1<Float>(randomUniform: particleCount)
  let randomTurn = middleDistribution.mask { $0 .< 0.1 } * cast(middleMask, elementsTo: Float.self)
  let turn = cast(highValues - 1, elementsTo: Float.self) * cast(1 - middleMask, elementsTo: Float.self) + randomTurn
  headings += (turn * moveAngle)
  positions += headings.angleToVector() * moveStep

  // Deposit
  // TODO: This wrapping around negative values needs to be fixed.
  let depositIndices = abs(cast(positions, elementsTo: Int32.self)) % gridShapeR2
  let deposits: TensorR2<Float> = scatter(number: 1.0, into: Shape2(gridSize, gridSize), indices: depositIndices)
  currentGrid += deposits

  // Diffuse
  currentGrid = pool(x: currentGrid, size: (3, 3), strides: (1, 1), pad: .same, mode: .averagePadding)
  currentGrid = currentGrid * evaporationRate
  
  grid[1 - phase] = currentGrid
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

