//******************************************************************************
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
    size: ImageSize,
    mode: FractalCalculationMode
) -> Tensor2 {
    let size2 = (r: size[0], c: size[1])

    let rFirst = Complex<Float>(range.start.real, 0), rLast = Complex<Float>(range.end.real, 0)
    let iFirst = Complex<Float>(0, range.start.imaginary), iLast = Complex<Float>(0, range.end.imaginary)

    let Xr = repeating(array(from: rFirst, to: rLast, (1, size2.c)), size2)
    let Xi = repeating(array(from: iFirst, to: iLast, (size2.r, 1)), size2)
    var Z = Xr + Xi

    var divergence = full(size, iterations)
    let start = Date()
    switch mode {
    case .direct:
        for i in 0..<iterations {
            Z = multiply(Z, Z, add: C)
            divergence[abs(Z) .> tolerance] = min(divergence, i)
        }
    case .parallelMap:
        // pmap(&Z, &divergence) { Z, divergence in
        //     print("\(Context.currentQueue.name)", Z.storageBase, divergence.storageBase)
        //     for i in 0..<1 {
        //         Z = multiply(Z, Z, add: C)
        //         divergence[abs(Z) .> tolerance] = min(divergence, i)
        //     }
        // }
        fatalError("Parallel map version not implemented.")
    case .kernel:
    #if canImport(SwiftRTCuda)
        let queue = currentQueue
        _ = divergence.withMutableTensor(using: queue) { d, dDesc in
            Z.withTensor(using: queue) {z, zDesc in
                withUnsafePointer(to: tolerance) { t in
                    withUnsafePointer(to: C) { c in
                        srtJulia(z, zDesc, d, dDesc, t, c, iterations, queue.stream)
                    }
                }
            }
        }

        // read on cpu to ensure gpu kernel is complete
        _ = divergence.read()
    #else
        for i in 0..<iterations {
            Z = multiply(Z, Z, add: C)
            divergence[abs(Z) .> tolerance] = min(divergence, i)
        }
    #endif
    }
    print("elapsed \(String(format: "%.3f", Date().timeIntervalSince(start))) seconds")

    return divergence
}
