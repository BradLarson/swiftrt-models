//**************************************************************************************************
// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
import Foundation
import SwiftRT

#if canImport(SwiftRTCuda)
  import SwiftRTCuda
#endif

func juliaSet(
  iterations: Int,
  constant C: Complex<Float>,
  tolerance: Float,
  range: ComplexRange,
  size imageSize: ImageSize,
  mode: FractalCalculationMode
) -> Tensor2 {
  let size = (r: imageSize[0], c: imageSize[1])
  print("rows: \(size.r), cols: \(size.c), iterations: \(iterations)")

  // repeat rows of real range, columns of imaginary range, and combine
  let rFirst = Complex<Float>(range.start.real, 0)
  let rLast = Complex<Float>(range.end.real, 0)
  let iFirst = Complex<Float>(0, range.start.imaginary)
  let iLast = Complex<Float>(0, range.end.imaginary)

  var Z =
    repeating(array(from: rFirst, to: rLast, shape: (1, size.c)), shape: size)
    + repeating(array(from: iFirst, to: iLast, shape: (size.r, 1)), shape: size)
  var divergence = mode == .kernel ? zeros(shape: size) : full(shape: size, iterations)

  //----------------------------------
  // perform the test
  let start = Date()
  switch mode {
  case .cachedMemory:
    // this is an initiali proof of concept that needs to have a prettier api
    let queue = currentQueue
    Z = Z.shared()
    var d = divergence.shared()
    var min_di = Tensor(like: d)
    var abs_Z = TensorR2<Float>(shape: Z.shape, order: Z.order)
    var gt_absZt2 = TensorR2<Bool>(shape: Z.shape, order: Z.order)
    let t2 = tolerance * tolerance

    for i in 0..<iterations {
      queue.multiply(Z, Z, add: C, &Z)
      queue.min(d, Float(i), &min_di)
      queue.abs2(Z, &abs_Z)
      queue.greater(abs_Z, t2, &gt_absZt2)
      queue.replace(d, min_di, gt_absZt2, &d)
    }
    queue.waitForCompletion()

  case .direct:
    for i in 0..<iterations {
      Z = multiply(Z, Z, add: C)
      divergence[abs(Z) .> tolerance] = min(divergence, i)
    }

  case .parallelMap:
    if currentDevice.index == 0 {
      usingSyncQueue {
        pmap(Z, &divergence) { Z, divergence in
          for i in 0..<iterations {
            Z = multiply(Z, Z, add: C)
            divergence[abs(Z) .> tolerance] = min(divergence, i)
          }
        }
      }
    } else {
      fatalError("GPU Parallel map version not implemented.")
    }
  case .kernel:
    if currentDevice.index == 0 {
      usingSyncQueue {
        pmap(Z, &divergence, limitedBy: .compute) {
          juliaCpuKernel(Z: $0, divergence: &$1, C, tolerance, Float(iterations))
        }
      }
    } else {
      #if canImport(SwiftRTCuda)
        let queue = currentQueue
        print("running cuda srtJuliaFlat kernel")
        _ = withUnsafePointer(to: C) { pC in
          srtJuliaFlat(
            Complex<Float>.type,
            Z.deviceRead(using: queue),
            pC,
            tolerance,
            iterations,
            divergence.count,
            divergence.deviceReadWrite(using: queue),
            queue.stream)
        }

        // this is only needed to make sure the work is done for
        // perf measurements
        queue.waitForCompletion()
      #endif
    }
  }
  print("JuliaSet elapsed \(String(format: "%.7f", Date().timeIntervalSince(start))) seconds")

  return divergence
}

//==================================================================================================
// user defined element wise function
@inlinable public func juliaCpuKernel<E>(
  Z: TensorR2<Complex<E>>,
  divergence: inout TensorR2<E>,
  _ c: Complex<E>,
  _ tolerance: E,
  _ iterations: E.Value
) where E: StorageElement & BinaryFloatingPoint, E.Value: BinaryFloatingPoint {
  let message =
    "julia(Z: \(Z.name), divergence: \(divergence.name), "
    + "constant: \(c), tolerance: \(tolerance)"

  kernel(Z, &divergence, message) { zval, _ in
    var z = zval
    var i = E.Value.zero
    while abs(z) <= tolerance && i < iterations {
      z = z * z + c
      i += 1
    }
    return i
  }
}
