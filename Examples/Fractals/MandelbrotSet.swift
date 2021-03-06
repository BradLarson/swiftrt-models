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

func mandelbrotSet(
  iterations: Int,
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

  let X =
    repeating(array(from: rFirst, to: rLast, shape: (1, size.c)), shape: size)
    + repeating(array(from: iFirst, to: iLast, shape: (size.r, 1)), shape: size)
  var Z = X
  var divergence = mode == .kernel ? empty(shape: size) : full(shape: size, iterations)

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
      queue.multiply(Z, Z, add: X, &Z)
      queue.min(d, Float(i), &min_di)
      queue.abs2(Z, &abs_Z)
      queue.greater(abs_Z, t2, &gt_absZt2)
      queue.replace(d, min_di, gt_absZt2, &d)
    }
    queue.waitForCompletion()

  case .direct:
    for i in 1..<iterations {
      divergence[abs(Z) .> tolerance] = min(divergence, i)
      Z = multiply(Z, Z, add: X)
    }

  case .parallelMap:
    if currentDevice.index == 0 {
      usingSyncQueue {
        // TODO: Have pmap take in X.
        // pmap(Z, X, &divergence) { Z, X, divergence in
        pmap(Z, &divergence) { Z, divergence in
          let X = Z
          for i in 1..<iterations {
            divergence[abs(Z) .> tolerance] = min(divergence, i)
            Z = multiply(Z, Z, add: X)
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
          mandelbrotCpuKernel(Z: $0, divergence: &$1, tolerance, Float(iterations))
        }
      }
    } else {
      #if canImport(SwiftRTCuda)
        let queue = currentQueue

        srtMandelbrotFlat(
          Complex<Float>.type,
          X.deviceRead(using: queue),
          tolerance,
          iterations,
          divergence.count,
          divergence.deviceReadWrite(using: queue),
          queue.stream)

        // this is only needed to make sure the work is done for
        // perf measurements
        queue.waitForCompletion()
      #endif
    }
  }

  print("MandelbrotSet elapsed \(String(format: "%.7f", Date().timeIntervalSince(start))) seconds")
  return divergence
}

//==================================================================================================
// user defined element wise function
@inlinable public func mandelbrotCpuKernel<E>(
  Z: TensorR2<Complex<E>>,
  divergence: inout TensorR2<E>,
  _ tolerance: E,
  _ iterations: E.Value
) where E == E.Value, E.Value: Real & Comparable {
  let message =
    "mandelbrot(Z: \(Z.name), divergence: \(divergence.name), "
    + "tolerance: \(tolerance), iterations: \(iterations))"

  kernel(Z, &divergence, message) { xval, _ in
    let x = xval
    var z = x
    var i = E.Value.zero
    while abs(z) <= tolerance && i < iterations {
      z = z * z + x
      i += 1
    }
    return i
  }
}
