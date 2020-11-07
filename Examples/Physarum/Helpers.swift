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

import SwiftRT

// TODO: Generalize argmin and upstream it.
@inlinable public func argmin3<E>(
    _ lhs: TensorR2<E>
) -> TensorR1<DeviceIndex> where E.Value: Numeric & Comparable & ComparableLimits {
  let slice1 = squeeze(lhs[0..<lhs.shape[0], 0], axis: 1)
  var slice2 = squeeze(lhs[0..<lhs.shape[0], 1], axis: 1)
  let slice3 = squeeze(lhs[0..<lhs.shape[0], 2], axis: 1)

  let mask1 = slice1 .> slice2
  let floatMask1 = cast(mask1, elementsTo: E.self)
  slice2 = slice2 * floatMask1 + slice1 * (1 - floatMask1)
  var indices = cast(mask1, elementsTo: DeviceIndex.self)
  let mask2 = cast(slice2 .> slice3, elementsTo: DeviceIndex.self)
  indices = indices * (1 - mask2) + 2 * mask2

  return indices
}

// TODO: Generalize argmax and upstream it.
@inlinable public func argmax3<E>(
    _ lhs: TensorR2<E>
) -> TensorR1<DeviceIndex> where E.Value: Numeric & Comparable & ComparableLimits {
  let slice1 = squeeze(lhs[0..<lhs.shape[0], 0], axis: 1)
  var slice2 = squeeze(lhs[0..<lhs.shape[0], 1], axis: 1)
  let slice3 = squeeze(lhs[0..<lhs.shape[0], 2], axis: 1)

  let mask1 = slice1 .< slice2
  let floatMask1 = cast(mask1, elementsTo: E.self)
  slice2 = slice2 * floatMask1 + slice1 * (1 - floatMask1)
  var indices = cast(mask1, elementsTo: DeviceIndex.self)
  let mask2 = cast(slice2 .< slice3, elementsTo: DeviceIndex.self)
  indices = indices * (1 - mask2) + 2 * mask2

  return indices
}

// TODO: Implement this as an actual SwiftRT operator in a better way than this.
extension Tensor where Element == Int32 {
  @inlinable public static func % (lhs: Self, rhs: Self) -> Self {
    return lhs - (cast(cast(lhs, elementsTo: Float.self) / cast(rhs, elementsTo: Float.self), elementsTo: TensorElement.self) * rhs)
  }
}

// TODO: Replace this with a more generalized function for scattering.
@inlinable public func scatter<E>(
  number: E.Value,
  into shape: Shape2,
  indices: TensorR2<DeviceIndex>
) -> TensorR2<E> where E.Value: Numeric {
  var result = zeros(shape: shape, E.self)
  for index in 0..<indices.shape[0] {
    let currentIndex = squeeze(indices[index], axis: 0)
    result[Int(currentIndex[0]), Int(currentIndex[1])] = number
  }

  return result
}

// TODO: Replace this with a more generalized and performant gather operator.
@inlinable public func gather<E>(
  from tensor: TensorR2<E>,
  indices: TensorR3<DeviceIndex>
) -> TensorR2<E> {
  let shape = Shape2(indices.shape[0], indices.shape[1])

  let squeezedIndices = reshape(indices, shape: (indices.shape[0] * indices.shape[1], indices.shape[2]))
  var values: [E.Value] = []
  for index in 0..<squeezedIndices.shape[0] {
    let currentIndex = squeeze(squeezedIndices[index], axis: 0)
    values.append(contentsOf: tensor[Int(currentIndex[1]), Int(currentIndex[0])].flatArray)
  }

  let result = TensorR2<E>(values, shape)
  return result
}
