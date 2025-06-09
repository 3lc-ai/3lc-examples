# Instance Refactoring Plan: BB Tools → Instance Tools Migration

## 🎯 Project Overview

**Goal**: Migrate the "per-bb properties" workflow to support "per-instance properties" covering both bounding boxes and segmentations under a unified "instances" concept.

**Current State**: The codebase contains hacky implementations mixing old BB code with new segmentation code, leading to duplication, hardcoded assumptions, and poor performance.

## 🔍 Current State Analysis

### Problems Identified:

1. **Mixed Terminology**: Codebase mixes "bounding box" (bb) and "segmentation" terminology with commented-out old BB code alongside new segmentation hacks.

2. **Hacky Implementation**: 
   - RLE decoding to extract bounding boxes from segmentations
   - Manual mask application to images  
   - Hardcoded schema creation
   - Duplicated logic across files

3. **Schema Inconsistencies**: Code tries to handle both:
   - Old format: `bbs.bb_list.label`
   - New format: `segmentations.rles` and `segmentations.instance_properties.label`

4. **Performance Issues**: 
   - Iterator approach in `extend_table_with_metrics.py` prevents multi-worker processing
   - Unnecessary `batched_bb_iterator` functionality
   - Single-threaded processing is prohibitive for large datasets

5. **Tool Name Mismatch**: CLI tool still called `augment-bb-table` despite handling instances

6. **Limited Flexibility**: 
   - Only works with specific schema structures
   - Hard-coded assumptions about RLE format
   - Poor abstraction of instance handling logic
   - **Requires labels**: Cannot handle anonymous/unlabeled instances from autosegmentation

## 📋 Complete Refactoring Plan

### Phase 1: Create Instance Abstraction Layer 🏗️

**Objective**: Create unified interface for working with both BBs and segmentations

**Key Components**:
- `InstanceHandler` class - abstract base for instance operations
- `BoundingBoxInstanceHandler` - handles traditional bounding boxes  
- `SegmentationInstanceHandler` - handles instance segmentation masks
- `InstanceDetector` - auto-detects instance type from table schema

**Deliverables**:
- `src/tlc_tools/instances/` package
- `InstanceHandler` abstract base class
- Concrete implementations for BB and segmentation types
- Unit tests for all instance handlers

**Benefits**:
- Unified API for cropping, metrics, and data access
- Eliminates hardcoded schema assumptions
- **Label-free operation**: Handle anonymous instances without labels
- Foundation for clean refactoring

### Phase 2: Refactor Core Components 🔧

**Objective**: Replace hacky implementations with clean abstraction-based code

**Key Changes**:

#### 2.1 Dataset Refactoring
- `BBCropDataset` → `InstanceCropDataset`
- Remove hardcoded RLE decoding logic
- Use `InstanceHandler` for all instance operations
- Clean up background generation logic

#### 2.2 Metrics Collection Overhaul
- **CRITICAL**: Replace iterator approach in `extend_table_with_metrics.py`
- Enable multi-worker processing capability
- Remove `batched_bb_iterator`, `bb_crop_iterator` functions
- Use standard DataLoader with `InstanceCropDataset`
- Implement proper batching without custom iterators
- **Label-free mode**: Extract embeddings from anonymous instances without labels

#### 2.3 Schema Handling
- Remove hardcoded schema paths
- Use `InstanceHandler` for schema detection and validation
- Support flexible instance property structures

**Deliverables**:
- Refactored `InstanceCropDataset` class
- Cleaned up `extend_table_with_metrics.py` with multi-worker support
- Removed iterator-based processing bottlenecks
- Updated `finetune_on_crops.py` to use new abstractions

### Phase 3: Update CLI and Documentation 📚

**Objective**: Update user-facing interfaces and maintain backward compatibility

**Key Changes**:
- Rename `augment-bb-table` → `augment-instances-table`
- Add deprecation warnings for old tool name
- Update README and help text
- Maintain full backward compatibility

**Deliverables**:
- New CLI command with updated naming
- Backward-compatible alias for old command
- Updated documentation and examples
- Migration guide for users

### Phase 4: Advanced Instance Support 🚀

**Objective**: Add comprehensive instance support and prepare for auto-labeling

**Key Features**:
- Support both mask-based and polygon-based segmentations
- Flexible instance property handling (not just labels)
- **Anonymous instance support**: Full pipeline for unlabeled instances
- Integration points for Phase 2 auto-labeling
- Performance optimizations

**Deliverables**:
- Polygon instance support
- Custom instance property schemas
- Auto-labeling integration points
- Performance benchmarks

## 🎯 Success Criteria

### Phase 1 Success:
- [ ] `InstanceHandler` abstraction covers all current BB/segmentation operations
- [ ] Zero hardcoded schema assumptions
- [ ] Unit tests pass for all instance types
- [ ] Clear separation of concerns

### Phase 2 Success:
- [ ] **Multi-worker processing enabled** (major performance win)
- [ ] All hacky RLE decoding removed
- [ ] Single codebase handles both BBs and segmentations
- [ ] No more commented-out duplicate code
- [ ] Existing functionality preserved

### Phase 3 Success:
- [ ] User-friendly CLI with updated naming
- [ ] Backward compatibility maintained
- [ ] Documentation reflects new capabilities
- [ ] Migration path is clear

### Phase 4 Success:
- [ ] Full instance segmentation ecosystem
- [ ] Ready for auto-labeling integration
- [ ] Performance optimized
- [ ] Extensible architecture

## 🚨 Critical Performance Fix

**Priority Issue**: The iterator approach in `extend_table_with_metrics.py` is a major bottleneck:
- Prevents multi-worker data loading
- Forces single-threaded processing
- Introduces unnecessary complexity with custom iterators

**Solution**: Replace with standard PyTorch DataLoader pattern using the new `InstanceCropDataset`.

## 🤝 Next Steps

1. **Review and Refine**: Take time to review this plan and make adjustments
2. **Start Phase 1**: Begin with `InstanceHandler` abstraction layer
3. **Collaborate on 3LC Details**: Leverage 3LC-specific implementation knowledge
4. **Iterative Development**: Build and test each phase incrementally

## 📝 Notes

- This plan maintains full backward compatibility
- Performance improvements are a major focus
- The abstraction layer enables future enhancements (auto-labeling, etc.)
- Multi-worker support is critical for production use
- **Label-free workflow**: Essential for SAM autosegmentation → feature extraction → auto-labeling pipeline

## 🎯 Label-Free Use Case

**Scenario**: Anonymous instances from SAM autosegmentation (like your Phase 1 results)

**Workflow**:
1. ✅ **Skip Training**: No labels available, can't train classifier
2. ✅ **Extract Features**: Crop anonymous instances and run through pre-trained model  
3. ✅ **Collect Embeddings**: Use existing model checkpoint for feature extraction
4. ✅ **Apply Dimensionality Reduction**: PaCMAP/UMAP for analysis dashboard
5. ✅ **Enable Auto-labeling**: Features become input for Phase 2 clustering/labeling

**Implementation Notes**:
- Detect missing labels gracefully (no crashes)
- Skip label-dependent operations (training, label mappings)
- Maintain all embedding and metrics functionality
- Support instance properties other than labels (score, area, etc.) 