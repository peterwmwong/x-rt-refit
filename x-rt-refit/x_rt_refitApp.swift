import AppKit
import Metal
import MetalKit
import SwiftUI

let WIDTH           = 512
let HEIGHT          = 512
var CAMERA_POSITION = SIMD4<Float32>(0.0, 0.0, -1.0, 1.0)

@main
struct x_rt_refitApp: App {
    @State var updateType: UpdateType? = nil
    @State var forceUpdate = 0
    var body: some Scene {
        WindowGroup {
            VStack {
                HStack {
                    Button("Refit") {
                        updateType = .refit
                        forceUpdate += 1
                    }
                    Button("Rebuild") {
                        updateType = .rebuild
                        forceUpdate += 1
                    }
                }
                MyMetalViewSwiftUIWrapper(updateType: updateType, forceUpdate: forceUpdate)
            }.frame(width: CGFloat(WIDTH), height: CGFloat(HEIGHT))
        }
    }
}

class MyMetalView: MTKView {
    lazy var cmdQueue            = device!.makeCommandQueue()!
    lazy var matrixScreenToWorld = makeMatrixScreenToWorld(window!.backingScaleFactor)
    lazy var modelAccelStruct    = ModelAccelerationStructure(device!, cmdQueue)
    lazy var pipeline            = makePipeline(device: device!, colorFormat: colorPixelFormat)

    init(device: MTLDevice?) {
        super.init(frame: .zero, device: device)
        isPaused              = true
        enableSetNeedsDisplay = true
        autoResizeDrawable    = true
    }
    
    override func draw(_ rect: CGRect) {
        let currentDrawable = currentDrawable!
        
        let attachment = currentRenderPassDescriptor!.colorAttachments[0]!
        attachment.texture     = currentDrawable.texture
        attachment.storeAction = .store
        attachment.loadAction  = .clear
        attachment.clearColor  = MTLClearColor(red: 0.0, green: 0.0, blue: 0.25, alpha: 0.0)
        
        let cmdBuf = cmdQueue.makeCommandBuffer()!
        let enc = cmdBuf.makeRenderCommandEncoder(descriptor: currentRenderPassDescriptor!)!
        enc.setRenderPipelineState(pipeline)
        enc.useHeap(modelAccelStruct.primAccelStructHeap, stages: .fragment)
        enc.setFragmentAccelerationStructure(modelAccelStruct.primAccelStruct, bufferIndex: 0)
        enc.setFragmentBytes(&matrixScreenToWorld, length: MemoryLayout<float4x4>.size, index: 1)
        enc.setFragmentBytes(&CAMERA_POSITION, length: MemoryLayout<Float32>.size * 4, index: 2)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
        
        cmdBuf.present(currentDrawable)
        cmdBuf.commit()
    }
    
    public func updateAccelerationStructuresAndDraw(_ update: UpdateType) {
        modelAccelStruct.update(update, device!, cmdQueue)
        draw()
    }
    
    @available(*, unavailable)
    required init(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
}
struct MyMetalViewSwiftUIWrapper: NSViewRepresentable {
    var updateType: UpdateType?
    var forceUpdate: Int
    func makeNSView(context: Context) -> MyMetalView {
        return MyMetalView(device: MTLCreateSystemDefaultDevice())
    }
    func updateNSView(_ view: MyMetalView, context: Context) {
        if let u = updateType {
            view.updateAccelerationStructuresAndDraw(u)
        }
    }
}

func makePipeline(device: MTLDevice, colorFormat: MTLPixelFormat) -> MTLRenderPipelineState {
    let lib = device.makeDefaultLibrary()!
    let desc = MTLRenderPipelineDescriptor()
    desc.vertexFunction                  = lib.makeFunction(name: "main_vertex")
    desc.fragmentFunction                = lib.makeFunction(name: "main_fragment")
    desc.colorAttachments[0].pixelFormat = colorFormat
    return try! device.makeRenderPipelineState(descriptor: desc)
}

func makeMatrixScreenToWorld(_ backingScaleFactor: CGFloat) -> float4x4 {
    let S: Float32       = 1.0 / tan((Float32.pi / 3.0) / 2.0)
    let N: Float32       = 0.1
    let F: Float32       = 100000.0
    let Z_RANGE: Float32 = F - N
    
    let matrixWorldToProjection = float4x4(rows: [
        SIMD4<Float32>(S,   0.0, 0.0,         0.0),
        SIMD4<Float32>(0.0, S,   0.0,         0.0),
        SIMD4<Float32>(0.0, 0.0, F / Z_RANGE, -N * F / Z_RANGE),
        SIMD4<Float32>(0.0, 0.0, 1.0,         0.0),
    ])
    
    let width  = Float32(WIDTH)  * Float32(backingScaleFactor)
    let height = Float32(HEIGHT) * Float32(backingScaleFactor)
    let matrixScreenToProjection = float4x4(rows: [
        SIMD4<Float32>(2.0 / width,   0.0,          0.0, -1.0),
        SIMD4<Float32>(0.0,          -2.0 / height, 0.0,  1.0),
        SIMD4<Float32>(0.0,           0.0,          1.0,  0.0),
        SIMD4<Float32>(0.0,           0.0,          0.0,  1.0),
    ])
    return matrixWorldToProjection.inverse * matrixScreenToProjection
}

func loadResource<T>(name: String) -> T {
    let url = Bundle.main.url(forResource: name, withExtension: "bin")!
    return try! Data(contentsOf: url).withUnsafeBytes { $0.assumingMemoryBound(to: T.self).first! }
}

func loadResourceAsMTLBuffer(device: MTLDevice, name: String) -> MTLBuffer {
    let url = Bundle.main.url(forResource: name, withExtension: "bin")!
    return try! Data(contentsOf: url).withUnsafeBytes {
        let buf = device.makeBuffer(length: $0.count, options: .storageModeShared)!
        buf.label = name
        buf.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        return buf
    }
}

enum UpdateType {
    case refit
    case rebuild
}

class ModelAccelerationStructure {
    var matrixModelToWorld:        float4x4
    let vertexBuffer:              MTLBuffer
    let indexBuffer:               MTLBuffer
    
    let primAccelStructDesc:       MTLPrimitiveAccelerationStructureDescriptor;
    let primAccelStructHeap:       MTLHeap
    let primAccelStruct:           MTLAccelerationStructure
    let primAccelStructRebuildBuf: MTLBuffer
    let primAccelStructRefitBuf:   MTLBuffer
    
    let triAccelStruct:            MTLAccelerationStructureTriangleGeometryDescriptor
    let triangleCount:             UInt32
    
    init(_ device: MTLDevice, _ cmdQueue: MTLCommandQueue) {
        triangleCount      = loadResource(name: "triangleCount")
        matrixModelToWorld = loadResource(name: "matrixModelToWorld")
        var matrixModelToWorldPacked = MTLPackedFloat4x3(columns: (
            MTLPackedFloat3Make(matrixModelToWorld.columns.0.x, matrixModelToWorld.columns.0.y, matrixModelToWorld.columns.0.z),
            MTLPackedFloat3Make(matrixModelToWorld.columns.1.x, matrixModelToWorld.columns.1.y, matrixModelToWorld.columns.1.z),
            MTLPackedFloat3Make(matrixModelToWorld.columns.2.x, matrixModelToWorld.columns.2.y, matrixModelToWorld.columns.2.z),
            MTLPackedFloat3Make(matrixModelToWorld.columns.3.x, matrixModelToWorld.columns.3.y, matrixModelToWorld.columns.3.z)
        ))
        vertexBuffer = loadResourceAsMTLBuffer(device: device, name: "vertexBuffer")
        indexBuffer = loadResourceAsMTLBuffer(device: device, name: "indexBuffer")
        
        // ========================================
        // Define Primitive Acceleration Structures
        // ========================================
        triAccelStruct = MTLAccelerationStructureTriangleGeometryDescriptor()
        triAccelStruct.vertexFormat       = .float3
        triAccelStruct.vertexBuffer       = vertexBuffer
        triAccelStruct.vertexBufferOffset = 0
        triAccelStruct.vertexStride       = MemoryLayout<Float32>.size * 3
        triAccelStruct.indexBuffer        = indexBuffer
        triAccelStruct.indexBufferOffset  = 0
        triAccelStruct.indexType          = .uint32
        triAccelStruct.triangleCount      = Int(triangleCount)
        triAccelStruct.opaque             = true
        triAccelStruct.label              = "triAccelStruct"
        triAccelStruct.transformationMatrixBuffer = device.makeBuffer(bytes: &matrixModelToWorldPacked, length: MemoryLayout<MTLPackedFloat4x3>.size)
        
        primAccelStructDesc = MTLPrimitiveAccelerationStructureDescriptor()
        primAccelStructDesc.geometryDescriptors = [triAccelStruct]
        let sizeAlign = device.heapAccelerationStructureSizeAndAlign(descriptor: primAccelStructDesc)
        var primAccelStructSizes = device.accelerationStructureSizes(descriptor: primAccelStructDesc)
        primAccelStructSizes.accelerationStructureSize = sizeAlign.size + sizeAlign.align
        
        primAccelStructRebuildBuf = device.makeBuffer(length: primAccelStructSizes.buildScratchBufferSize)!
        primAccelStructRefitBuf = device.makeBuffer(length: primAccelStructSizes.refitScratchBufferSize)!
        
        let primAccelStructHeapDesc = MTLHeapDescriptor()
        primAccelStructHeapDesc.storageMode = .private
        primAccelStructHeapDesc.size        = primAccelStructSizes.accelerationStructureSize
        primAccelStructHeap = device.makeHeap(descriptor: primAccelStructHeapDesc)!
        
        primAccelStruct = primAccelStructHeap.makeAccelerationStructure(size: sizeAlign.size)!
        let primAccelStructScratchBuf = device.makeBuffer(length: primAccelStructSizes.buildScratchBufferSize)!
        primAccelStructScratchBuf.label = "primAccelStructScratchBuf"
        
        // ======================================
        // Initiate build Acceleration Structures
        // ======================================
        let cmdBuf = cmdQueue.makeCommandBufferWithUnretainedReferences()!
        let enc = cmdBuf.makeAccelerationStructureCommandEncoder()!
        enc.useHeap(primAccelStructHeap)
        enc.useResource(vertexBuffer, usage: .read)
        enc.useResource(indexBuffer, usage: .read)
        enc.build(accelerationStructure: primAccelStruct, descriptor: primAccelStructDesc, scratchBuffer: primAccelStructScratchBuf, scratchBufferOffset: 0)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        assert(cmdBuf.status == .completed)
    }
    
    public func update(_ update: UpdateType, _ device: MTLDevice, _ cmdQueue: MTLCommandQueue) {
        matrixModelToWorld = float4x4(rows: [
            SIMD4<Float32>(1.0, 0.0, 0.0, 0.1),
            SIMD4<Float32>(0.0, 1.0, 0.0, 0.0),
            SIMD4<Float32>(0.0, 0.0, 1.0, 0.0),
            SIMD4<Float32>(0.0, 0.0, 0.0, 1.0),
        ]) * matrixModelToWorld;
        triAccelStruct.transformationMatrixBuffer!.contents().assumingMemoryBound(to: MTLPackedFloat4x3.self).pointee = MTLPackedFloat4x3(columns: (
            MTLPackedFloat3Make(matrixModelToWorld.columns.0.x, matrixModelToWorld.columns.0.y, matrixModelToWorld.columns.0.z),
            MTLPackedFloat3Make(matrixModelToWorld.columns.1.x, matrixModelToWorld.columns.1.y, matrixModelToWorld.columns.1.z),
            MTLPackedFloat3Make(matrixModelToWorld.columns.2.x, matrixModelToWorld.columns.2.y, matrixModelToWorld.columns.2.z),
            MTLPackedFloat3Make(matrixModelToWorld.columns.3.x, matrixModelToWorld.columns.3.y, matrixModelToWorld.columns.3.z)
        ))
    
        let cmdBuf = cmdQueue.makeCommandBuffer()!
        let enc = cmdBuf.makeAccelerationStructureCommandEncoder()!
        enc.useHeap(primAccelStructHeap)
        enc.useResource(vertexBuffer, usage: .read)
        enc.useResource(indexBuffer, usage: .read)
        switch update {
        case .refit:
            assert(false, "THIS DOES NOT WORK RIGHT NOW (BETA 5)");
            print("refitting...")
            enc.refit(
                sourceAccelerationStructure:      primAccelStruct,
                descriptor:                       primAccelStructDesc,
                destinationAccelerationStructure: nil,
                scratchBuffer:                    primAccelStructRefitBuf,
                scratchBufferOffset:              0
            )
        case .rebuild:
            print("rebuilding...")
            enc.build(
                accelerationStructure: primAccelStruct,
                descriptor:            primAccelStructDesc,
                scratchBuffer:         primAccelStructRebuildBuf,
                scratchBufferOffset:   0
            )
        }
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        print("done")
        assert(cmdBuf.status == .completed)
    }
}
