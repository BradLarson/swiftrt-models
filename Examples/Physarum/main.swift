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

let gridShapeR3 = repeating(array([Int32(gridSize), Int32(gridSize)], (1, 1, 2)), (particleCount, 3, 2))
let gridShapeR2 = repeating(array([Int32(gridSize), Int32(gridSize)], (1, 2)), (particleCount, 2))

extension Tensor where TensorElement.Value: Numeric {
  func mask(condition: (Tensor<Shape, TensorElement>) -> Tensor<Shape, Bool>) -> Tensor<Shape, TensorElement> {
    return Tensor(condition(self))
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
  let sensingIndices = abs(TensorR3<Int32>(sensingPosition)) % gridShapeR3

  // TODO: Gather
  //  let sensedValues = currentGrid.expandingShape(at: 2)
  //    .dimensionGathering(atIndices: sensingIndices).squeezingShape(at: 2)

  let sensedValues = array([-3.0, 0.0, 2.0,
                            1.0, 1.0, 1.0,
                            1.0, 2.0, 3.0,
                            3.0, 2.0, 1.0,
                            -1.0, -2.0, -3.0], (5, 3)) // PLACEHOLDER

  // Move
  //  let lowValues = argmin(sensedValues) // lowValues should be [0, 0-2, 0, 2, 2]
  //  let highValues = argmax(sensedValues) // highValues should be [2, 0-2, 2, 0, 0]
  let lowValues: TensorR1<Int32> = array([0, 1, 0, 2, 2])  // PLACEHOLDER
  let highValues: TensorR1<Int32> = array([2, 1, 2, 0, 0]) // PLACEHOLDER

  let middleMask = lowValues.mask { $0 .== 1 }
  let middleDistribution = TensorR1<Float>(randomUniform: particleCount)
  let randomTurn = middleDistribution.mask { $0 .< 0.1 } * TensorR1<Float>(middleMask)
  let turn = TensorR1<Float>(highValues - 1) * TensorR1<Float>(1 - middleMask) + randomTurn
  headings += (turn * moveAngle)
  positions += headings.angleToVector() * moveStep

  // Deposit
  // TODO: This wrapping around negative values needs to be fixed.
  let depositIndices = abs(TensorR2<Int32>(positions)) % gridShapeR2
  let deposits: TensorR2<Float> = scatter(number: 1.0, into: Shape2(gridSize, gridSize), indices: depositIndices)
  currentGrid += deposits

  // Diffuse

  // TODO: 3x3 average pool with 0-padding
  // currentGrid = avgPool2D(currentGrid, filterSize: (1, 3, 3, 1), strides: (1, 1, 1, 1), padding: .valid)

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

// TODO: argmin and argmax should take in comparable values and provide indices out.
// TODO: argmin and argmax should provide a squeezing axis parameter.
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

// TODO: Add this into main SwiftRT.
// TODO: Add an equality operator with the rhs being a scalar.
extension Tensor where TensorElement.Value: Equatable & StorageElement {
    /// Performs element-wise equality comparison and returns a
    /// tensor of Bool values
    @inlinable public static func .== (
        _ lhs: Self,
        _ rhs: TensorElement.Value
    ) -> Tensor<Shape, Bool> {
        var result = Tensor<Shape, Bool>(shape: lhs.shape, order: lhs.order)
        let expandedRHS = Tensor<Shape, TensorElement>(repeating: rhs, to: lhs.shape)
        currentQueue.equal(lhs, expandedRHS, &result)
        return result
    }
}

// TODO: Implement this as an actual SwiftRT operator in a better way than this.
extension Tensor where Element == Int32 {
  @inlinable public static func % (lhs: Self, rhs: Self) -> Self {
    return lhs - (Tensor(Tensor<Shape, Float>(lhs) / Tensor<Shape, Float>(rhs)) * rhs)
  }
}

// TODO: Replace this with a more generalized function for scattering.
@inlinable public func scatter<E>(
  number: E.Value,
  into shape: Shape2,
  indices: TensorR2<DeviceIndex>
) -> TensorR2<E> where E.Value: Numeric {
  var result = zeros(shape, E.self)
  for index in 0..<indices.shape[0] {
    let currentIndex = squeeze(indices[index], axis: 0)
    result[Int(currentIndex[0]), Int(currentIndex[1])] = number
  }

  return result
}
