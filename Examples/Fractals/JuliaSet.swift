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
    size imageSize: ImageSize,
    mode: FractalCalculationMode
) -> Tensor2 {
    let size = (r: 30, c: 30) // (r: imageSize[0], c: imageSize[1])

    let rFirst = Complex<Float>(range.start.real, 0)
    let rLast  = Complex<Float>(range.end.real, 0)
    let iFirst = Complex<Float>(0, range.start.imaginary)
    let iLast  = Complex<Float>(0, range.end.imaginary)

    // repeat rows of real range, columns of imaginary range, and combine
    var Z = repeating(array(from: rFirst, to: rLast, (1, size.c)), size) +
            repeating(array(from: iFirst, to: iLast, (size.r, 1)), size)
    var divergence = mode == .kernel ? zeros(size) : full(size, iterations)

    print("rows: \(size.r), cols: \(size.c), iterations: \(iterations)")

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
        print("running cuda kernel")
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
        print(divergence)
    #else
        for i in 0..<iterations {
            Z = multiply(Z, Z, add: C)
            divergence[abs(Z) .> tolerance] = min(divergence, i)
        }
    #endif
    }
    print("JuliaSet elapsed \(String(format: "%.7f", Date().timeIntervalSince(start))) seconds")

    return divergence
}
