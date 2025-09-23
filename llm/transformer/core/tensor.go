// package core 提供了类似 NumPy/PyTorch 的基础多维数组（张量）操作。
// 它支持张量的创建、数学运算、形状变换和一些基本的机器学习函数。
// 注意：这是一个教学性质的简单实现，并未对所有操作进行极致性能优化（例如，未使用 BLAS/LAPACK）。
package core

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"
)

// init 在包加载时执行，用于设置全局随机数生成器的种子。
// 这可以避免在每次调用随机函数时都重新播种，从而提高性能和随机性。
func init() {
	rand.Seed(time.Now().UnixNano())
}

// =================================================================================================
// == Tensor 结构体定义 ==
// =================================================================================================

// Tensor 是一个多维数组（张量）的核心结构。
type Tensor struct {
	data    []float64 // 使用一维切片以行优先（Row-major）顺序存储多维数据。
	shape   []int     // 描述张量在每个维度上的大小。例如, [2, 3] 表示一个2x3的矩阵。
	strides []int     // 步长，用于将多维索引映射到一维 data 切片中的位置。
	size    int       // 张量中元素的总数，等于 shape 各维度大小的乘积。

	// --- 自动求导相关字段 (当前版本暂未实现功能) ---
	RequiresGrad bool    // 如果为 true，则在计算中追踪该张量的梯度。
	Grad         *Tensor // 存储该张量的梯度，其形状与原张量相同。
}

// =================================================================================================
// == 构造函数 (Constructors) ==
// =================================================================================================

// NewTensor 创建一个指定形状的零值张量。
// 形状 (shape) 中的每个维度必须是正整数。
func NewTensor(shape ...int) *Tensor {
	if len(shape) == 0 {
		// 允许创建标量，形状为[1]，但 size 为1
		shape = []int{1}
	}

	size := 1
	for _, dim := range shape {
		if dim <= 0 {
			panic(fmt.Sprintf("invalid dimension: a dimension must be a positive integer, but got %d", dim))
		}
		size *= dim
	}

	// 计算步长（行优先存储）
	// 例如, shape [2, 3, 4] 的步长为 [12, 4, 1]
	// 访问 (i, j, k) 元素的索引是 i*12 + j*4 + k*1
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}

	return &Tensor{
		data:    make([]float64, size),
		shape:   shape,
		strides: strides,
		size:    size,
	}
}

// NewTensorFromData 根据给定的数据和形状创建一个新的张量。
// 数据长度必须与形状定义的总大小相匹配。
func NewTensorFromData(data []float64, shape ...int) *Tensor {
	t := NewTensor(shape...)
	if len(data) != t.size {
		panic(fmt.Sprintf("data length mismatch: provided data has %d elements, but shape %v requires %d", len(data), shape, t.size))
	}
	copy(t.data, data)
	return t
}

// Zeros 创建一个指定形状的全零张量，是 NewTensor 的别名。
func Zeros(shape ...int) *Tensor {
	return NewTensor(shape...)
}

// Ones 创建一个指定形状的全一张量。
func Ones(shape ...int) *Tensor {
	t := NewTensor(shape...)
	for i := range t.data {
		t.data[i] = 1.0
	}
	return t
}

// Eye 创建一个 N x N 的单位矩阵。
func Eye(n int) *Tensor {
	if n <= 0 {
		panic("matrix size for Eye must be positive")
	}
	t := NewTensor(n, n)
	for i := 0; i < n; i++ {
		// (i, i) 的位置索引是 i * strides[0] + i * strides[1]
		// 对于 2D 矩阵，strides[0]=n, strides[1]=1, 所以索引是 i*n + i
		t.data[i*n+i] = 1.0
	}
	return t
}

// Rand 创建一个在 [0.0, 1.0) 区间内均匀分布的随机张量。
func Rand(shape ...int) *Tensor {
	t := NewTensor(shape...)
	for i := range t.data {
		t.data[i] = rand.Float64()
	}
	return t
}

// Randn 创建一个服从标准正态分布（均值为0，方差为1）的随机张量。
// 使用 Box-Muller 变换生成。
func Randn(shape ...int) *Tensor {
	t := NewTensor(shape...)
	for i := 0; i < len(t.data); i += 2 {
		// Box-Muller transform
		u1 := rand.Float64()
		u2 := rand.Float64()
		z0 := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
		z1 := math.Sqrt(-2*math.Log(u1)) * math.Sin(2*math.Pi*u2)

		t.data[i] = z0
		if i+1 < len(t.data) {
			t.data[i+1] = z1
		}
	}
	return t
}

// Arange 创建一个在 [start, stop) 区间内以指定步长 `step` 生成的一维张量。
func Arange(start, stop, step float64) *Tensor {
	if step == 0 {
		panic("step cannot be zero")
	}
	size := int(math.Ceil((stop - start) / step))
	if size <= 0 {
		return NewTensor(0) // Return empty tensor
	}
	t := NewTensor(size)
	for i := 0; i < size; i++ {
		t.data[i] = start + float64(i)*step
	}
	return t
}

// Linspace 创建一个包含 `num` 个元素的一维张量，元素值在 [start, stop] 区间内均匀分布。
func Linspace(start, stop float64, num int) *Tensor {
	if num <= 0 {
		panic("number of samples must be positive")
	}
	t := NewTensor(num)
	if num == 1 {
		t.data[0] = start
		return t
	}

	step := (stop - start) / float64(num-1)
	for i := 0; i < num; i++ {
		t.data[i] = start + float64(i)*step
	}
	return t
}

// =================================================================================================
// == 属性与访问器 (Properties & Accessors) ==
// =================================================================================================

// Shape 返回张量形状的副本。
func (t *Tensor) Shape() []int {
	shapeCopy := make([]int, len(t.shape))
	copy(shapeCopy, t.shape)
	return shapeCopy
}

// Strides 返回张量步长的副本。
func (t *Tensor) Strides() []int {
	stridesCopy := make([]int, len(t.strides))
	copy(stridesCopy, t.strides)
	return stridesCopy
}

// Size 返回张量中元素的总数。
func (t *Tensor) Size() int {
	return t.size
}

// Ndim 返回张量的维度数。
func (t *Tensor) Ndim() int {
	return len(t.shape)
}

// Data 返回底层数据切片的副本。
func (t *Tensor) Data() []float64 {
	dataCopy := make([]float64, t.size)
	copy(dataCopy, t.data)
	return dataCopy
}

// flatIndex 将多维索引转换为底层一维数据切片中的索引。
func (t *Tensor) flatIndex(indices []int) int {
	if len(indices) != len(t.shape) {
		panic(fmt.Sprintf("index dimension mismatch: expected %d dimensions, but got %d", len(t.shape), len(indices)))
	}

	index := 0
	for i, idx := range indices {
		if idx < 0 || idx >= t.shape[i] {
			panic(fmt.Sprintf("index out of bounds: index %d is out of bounds for dimension %d with size %d", idx, i, t.shape[i]))
		}
		index += idx * t.strides[i]
	}
	return index
}

// Get 获取指定多维索引处的元素值。
func (t *Tensor) Get(indices ...int) float64 {
	return t.data[t.flatIndex(indices)]
}

// Set 设置指定多维索引处的元素值。
func (t *Tensor) Set(value float64, indices ...int) {
	t.data[t.flatIndex(indices)] = value
}

// SetRow sets a row in a 2D tensor.
func (t *Tensor) SetRow(row *Tensor, index int) {
	if t.Ndim() != 2 || row.Ndim() != 2 || row.shape[0] != 1 {
		panic("SetRow is only implemented for setting a row in a 2D tensor with a 2D row tensor")
	}
	if t.shape[1] != row.shape[1] {
		panic("row length mismatch")
	}
	if index < 0 || index >= t.shape[0] {
		panic("row index out of bounds")
	}

	offset := index * t.strides[0]
	copy(t.data[offset:offset+t.shape[1]], row.data)
}

// Slice extracts a slice from the tensor.
func (t *Tensor) Slice(indices ...int) *Tensor {
	// This is a simplified slice that only supports slicing along the first dimension.
	if len(indices) != 2 {
		panic("Slice only supports 2 arguments for the first dimension")
	}
	start, end := indices[0], indices[1]
	if start < 0 || end > t.shape[0] || start > end {
		panic("Invalid slice indices")
	}

	newShape := make([]int, t.Ndim())
	copy(newShape, t.shape)
	newShape[0] = end - start

	newData := t.data[start*t.strides[0] : end*t.strides[0]]
	return NewTensorFromData(newData, newShape...)
}

// =================================================================================================
// == 元素级数学运算 (Element-wise Math Operations) ==
// =================================================================================================

// Add performs element-wise addition. Supports broadcasting. Returns a new tensor.
func (t *Tensor) Add(other *Tensor) *Tensor {
	return t._elementWiseOp(other, func(a, b float64) float64 { return a + b }, false)
}

// Sub performs element-wise subtraction. Supports broadcasting. Returns a new tensor.
func (t *Tensor) Sub(other *Tensor) *Tensor {
	return t._elementWiseOp(other, func(a, b float64) float64 { return a - b }, false)
}

// Sub_ performs in-place element-wise subtraction.
func (t *Tensor) Sub_(other *Tensor) {
	t._elementWiseOp(other, func(a, b float64) float64 { return a - b }, true)
}

// Mul performs element-wise multiplication. Supports broadcasting. Returns a new tensor.
func (t *Tensor) Mul(other *Tensor) *Tensor {
	return t._elementWiseOp(other, func(a, b float64) float64 { return a * b }, false)
}

// Div performs element-wise division. Supports broadcasting. Returns a new tensor.
func (t *Tensor) Div(other *Tensor) *Tensor {
	return t._elementWiseOp(other, func(a, b float64) float64 {
		if b == 0 {
			panic("division by zero")
		}
		return a / b
	}, false)
}

// AddScalar adds a scalar to each element of the tensor. Returns a new tensor.
func (t *Tensor) AddScalar(scalar float64) *Tensor {
	return t._elementWiseScalarOp(scalar, func(a, b float64) float64 { return a + b }, false)
}

// MulScalar multiplies each element of the tensor by a scalar. Returns a new tensor.
func (t *Tensor) MulScalar(scalar float64) *Tensor {
	return t._elementWiseScalarOp(scalar, func(a, b float64) float64 { return a * b }, false)
}

// DivScalar divides each element of the tensor by a scalar. Returns a new tensor.
func (t *Tensor) DivScalar(scalar float64) *Tensor {
	if scalar == 0 {
		panic("scalar division by zero")
	}
	return t._elementWiseScalarOp(scalar, func(a, b float64) float64 { return a / b }, false)
}

// Power computes the exponent of each element of the tensor. Returns a new tensor.
func (t *Tensor) Power(exponent float64) *Tensor {
	return t._elementWiseScalarOp(exponent, func(a, b float64) float64 { return math.Pow(a, b) }, false)
}

// Abs computes the absolute value of each element of the tensor. Returns a new tensor.
func (t *Tensor) Abs() *Tensor {
	return t._elementWiseOp(nil, func(a, b float64) float64 { return math.Abs(a) }, false)
}

// Exp computes the exponential of each element of the tensor. Returns a new tensor.
func (t *Tensor) Exp() *Tensor {
	return t._elementWiseOp(nil, func(a, b float64) float64 { return math.Exp(a) }, false)
}

// Log computes the natural logarithm of each element of the tensor. Returns a new tensor.
func (t *Tensor) Log() *Tensor {
	return t._elementWiseOp(nil, func(a, b float64) float64 { return math.Log(a) }, false)
}

// Sqrt computes the square root of each element of the tensor. Returns a new tensor.
func (t *Tensor) Sqrt() *Tensor {
	return t._elementWiseOp(nil, func(a, b float64) float64 { return math.Sqrt(a) }, false)
}

// _elementWiseOp is the generalized element-wise operation.
func (t *Tensor) _elementWiseOp(other *Tensor, op func(a, b float64) float64, inPlace bool) *Tensor {
	var result *Tensor
	if inPlace {
		result = t
	} else {
		result = t.Clone()
	}

	if other == nil {
		// Unary operation
		for i := 0; i < t.size; i++ {
			result.data[i] = op(t.data[i], 0) // 0 is a dummy value for unary ops
		}
		return result
	}

	if t.sameShape(other) {
		for i := 0; i < t.size; i++ {
			result.data[i] = op(t.data[i], other.data[i])
		}
		return result
	}

	// Broadcasting logic (simplified)
	resultShape, broadcastA, broadcastB := broadcastShapes(t.shape, other.shape)
	if !inPlace {
		result = NewTensor(resultShape...)
	}

	aIter := newBroadcastIterator(t, broadcastA)
	bIter := newBroadcastIterator(other, broadcastB)

	for i := 0; i < result.size; i++ {
		result.data[i] = op(aIter.Next(), bIter.Next())
	}

	return result
}

// _elementWiseScalarOp is the generalized element-wise scalar operation.
func (t *Tensor) _elementWiseScalarOp(scalar float64, op func(a, b float64) float64, inPlace bool) *Tensor {
	var result *Tensor
	if inPlace {
		result = t
	} else {
		result = t.Clone()
	}

	for i := 0; i < t.size; i++ {
		result.data[i] = op(t.data[i], scalar)
	}
	return result
}

// =================================================================================================
// == 矩阵乘法 (Matrix Multiplication) ==
// =================================================================================================

// Dot 执行点积运算。行为取决于张量的维度：
// - 1D x 1D: 向量内积，返回一个标量张量 (shape [1])。
// - 2D x 2D: 标准矩阵乘法。
// - ND x ND: 批量矩阵乘法 (要求批次维度匹配)。
func (t *Tensor) Dot(other *Tensor) *Tensor {
	// 1D · 1D -> 向量内积
	if t.Ndim() == 1 && other.Ndim() == 1 {
		if t.shape[0] != other.shape[0] {
			panic(fmt.Sprintf("shape mismatch for dot product: %v and %v", t.shape, other.shape))
		}
		sum := 0.0
		for i := 0; i < t.shape[0]; i++ {
			sum += t.data[i] * other.data[i]
		}
		return NewTensorFromData([]float64{sum}, 1)
	}

	// 2D · 2D -> 矩阵乘法
	if t.Ndim() == 2 && other.Ndim() == 2 {
		return t.matmul2D(other)
	}

	// 3D · 3D -> 批量矩阵乘法
	if t.Ndim() == 3 && other.Ndim() == 3 {
		return t.batchMatMul(other)
	}

	// TODO: 实现更通用的 N-D 矩阵乘法
	panic(fmt.Sprintf("unsupported dot product for shapes %v and %v", t.shape, other.shape))
}

// matmul2D 是一个优化的 2D 矩阵乘法实现。
func (t *Tensor) matmul2D(other *Tensor) *Tensor {
	if t.shape[1] != other.shape[0] {
		panic(fmt.Sprintf("shape mismatch for matrix multiplication: %v and %v", t.shape, other.shape))
	}

	rowsA := t.shape[0]
	colsA := t.shape[1] // a.k.a. innerDim
	colsB := other.shape[1]

	result := NewTensor(rowsA, colsB)

	// 直接访问底层数据以获得最佳性能，避免 Get/Set 开销。
	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsB; j++ {
			sum := 0.0
			// 优化：预先计算 t 在当前行的起始偏移量
			offsetA := i * t.strides[0]
			for k := 0; k < colsA; k++ {
				// a[i, k] * b[k, j]
				sum += t.data[offsetA+k] * other.data[k*other.strides[0]+j]
			}
			result.data[i*result.strides[0]+j] = sum
		}
	}

	return result
}

// batchMatMul 执行批量矩阵乘法。
// 输入张量形状应为 [batch, rows, cols]。
func (t *Tensor) batchMatMul(other *Tensor) *Tensor {
	if t.shape[0] != other.shape[0] {
		panic(fmt.Sprintf("batch size mismatch for batch matrix multiplication: %d vs %d", t.shape[0], other.shape[0]))
	}
	if t.shape[2] != other.shape[1] {
		panic(fmt.Sprintf("inner dimension mismatch for batch matrix multiplication: %d vs %d", t.shape[2], other.shape[1]))
	}

	batchSize := t.shape[0]
	rowsA := t.shape[1]
	innerDim := t.shape[2]
	colsB := other.shape[2]

	result := NewTensor(batchSize, rowsA, colsB)

	// 计算每个 2D 矩阵在底层数据中的大小
	matSizeA := t.shape[1] * t.shape[2]
	matSizeB := other.shape[1] * other.shape[2]
	matSizeResult := result.shape[1] * result.shape[2]

	for b := 0; b < batchSize; b++ {
		// 计算当前批次矩阵的起始偏移量
		offsetA := b * matSizeA
		offsetB := b * matSizeB
		offsetR := b * matSizeResult

		for i := 0; i < rowsA; i++ {
			for j := 0; j < colsB; j++ {
				sum := 0.0
				for k := 0; k < innerDim; k++ {
					// a[b, i, k] * b[b, k, j]
					sum += t.data[offsetA+i*innerDim+k] * other.data[offsetB+k*colsB+j]
				}
				// result[b, i, j]
				result.data[offsetR+i*colsB+j] = sum
			}
		}
	}

	return result
}

// =================================================================================================
// == 形状操作 (Shape Manipulation) ==
// =================================================================================================

// Reshape 在不改变数据的情况下，为张量赋予新的形状。
// 新形状的总元素数必须与原形状相同。
// newShape 中最多可以有一个维度为 -1，它将被自动推断。
func (t *Tensor) Reshape(newShape ...int) *Tensor {
	newSize := 1
	autoIdx := -1

	for i, dim := range newShape {
		if dim == -1 {
			if autoIdx != -1 {
				panic("multiple -1 dimensions in reshape is not allowed")
			}
			autoIdx = i
		} else if dim > 0 {
			newSize *= dim
		} else {
			panic(fmt.Sprintf("invalid dimension size in reshape: %d", dim))
		}
	}

	if autoIdx != -1 {
		if t.size%newSize != 0 {
			panic(fmt.Sprintf("cannot reshape tensor of size %d into shape with inferred dimension", t.size))
		}
		newShape[autoIdx] = t.size / newSize
	} else {
		finalNewSize := 1
		for _, dim := range newShape {
			finalNewSize *= dim
		}
		if finalNewSize != t.size {
			panic(fmt.Sprintf("cannot reshape tensor of size %d into shape %v (new size %d)", t.size, newShape, finalNewSize))
		}
	}

	// 因为数据布局不变，只需创建一个新的 Tensor 结构体共享底层数据。
	// 为简单起见，这里创建新数据副本，更高级的实现会共享内存并创建 "view"。
	result := NewTensor(newShape...)
	copy(result.data, t.data)
	return result
}

// Transpose 交换张量的维度。
// 如果不提供 `axes` 参数，则反转所有维度（例如，(a,b,c) -> (c,b,a)）。
// 否则，`axes` 必须是 `[0, 1, ..., Ndim-1]` 的一个排列。
func (t *Tensor) Transpose(axes ...int) *Tensor {
	dims := t.Ndim()

	// 默认行为：反转所有轴
	if len(axes) == 0 {
		axes = make([]int, dims)
		for i := 0; i < dims; i++ {
			axes[i] = dims - 1 - i
		}
	}

	if len(axes) != dims {
		panic(fmt.Sprintf("axes length mismatch: expected %d axes, but got %d", dims, len(axes)))
	}

	// 验证轴的有效性
	seen := make(map[int]bool)
	for _, axis := range axes {
		if axis < 0 || axis >= dims {
			panic(fmt.Sprintf("invalid axis %d for a %d-dimensional tensor", axis, dims))
		}
		if seen[axis] {
			panic(fmt.Sprintf("repeated axis %d in transpose", axis))
		}
		seen[axis] = true
	}

	// 计算新形状和新步长
	newShape := make([]int, dims)
	for i, axis := range axes {
		newShape[i] = t.shape[axis]
	}

	result := NewTensor(newShape...)

	// 迭代地复制数据
	indices := make([]int, dims)
	t.transposeRecursive(result, indices, 0, axes)

	return result
}

// transposeRecursive 是 Transpose 的递归辅助函数。
func (t *Tensor) transposeRecursive(result *Tensor, indices []int, depth int, axes []int) {
	if depth == t.Ndim() {
		// 创建转置后的索引
		newIndices := make([]int, len(indices))
		for i, axis := range axes {
			newIndices[i] = indices[axis]
		}

		value := t.Get(indices...)
		result.Set(value, newIndices...)
		return
	}

	for i := 0; i < t.shape[depth]; i++ {
		indices[depth] = i
		t.transposeRecursive(result, indices, depth+1, axes)
	}
}

// Squeeze 移除张量中大小为 1 的维度。
// 如果不提供 `axes`，则移除所有大小为 1 的维度。
// 如果提供了 `axes`，则只尝试移除指定的维度。
func (t *Tensor) Squeeze(axes ...int) *Tensor {
	newShape := []int{}

	if len(axes) == 0 {
		for _, dim := range t.shape {
			if dim != 1 {
				newShape = append(newShape, dim)
			}
		}
	} else {
		squeezeAxes := make(map[int]bool)
		for _, axis := range axes {
			// 处理负数轴
			if axis < 0 {
				axis += t.Ndim()
			}
			if axis < 0 || axis >= t.Ndim() {
				panic(fmt.Sprintf("invalid axis %d for squeeze", axis))
			}
			if t.shape[axis] != 1 {
				panic(fmt.Sprintf("cannot squeeze axis %d because its size is %d, not 1", axis, t.shape[axis]))
			}
			squeezeAxes[axis] = true
		}

		for i, dim := range t.shape {
			if !squeezeAxes[i] {
				newShape = append(newShape, dim)
			}
		}
	}

	// 如果所有维度都被压缩，结果是一个标量，形状为 [1]
	if len(newShape) == 0 {
		newShape = []int{1}
	}

	return t.Reshape(newShape...)
}

// ExpandAs broadcasts the tensor to a new shape.
// The new shape must be compatible for broadcasting.
// This is a simplified, data-copying implementation. A proper implementation would use strides.
func (t *Tensor) ExpandAs(shape ...int) *Tensor {
	if t.Ndim() > len(shape) {
		panic(fmt.Sprintf("new shape must have at least as many dimensions as the original: %d vs %d", len(shape), t.Ndim()))
	}

	// Check if shapes are compatible for broadcasting
	for i := 1; i <= t.Ndim(); i++ {
		srcDim := t.shape[t.Ndim()-i]
		tgtDim := shape[len(shape)-i]
		if srcDim != tgtDim && srcDim != 1 {
			panic(fmt.Sprintf("cannot broadcast shape %v to %v", t.shape, shape))
		}
	}

	result := NewTensor(shape...)
	resultIndices := make([]int, len(shape))
	srcIndices := make([]int, t.Ndim())

	for i := 0; i < result.size; i++ {
		temp := i
		for j := len(shape) - 1; j >= 0; j-- {
			resultIndices[j] = temp % shape[j]
			temp /= shape[j]
		}

		// Align from the right (trailing dimensions)
		for j := 1; j <= t.Ndim(); j++ {
			srcDimIdx := t.Ndim() - j
			tgtDimIdx := len(shape) - j

			if t.shape[srcDimIdx] == shape[tgtDimIdx] {
				srcIndices[srcDimIdx] = resultIndices[tgtDimIdx]
			} else {
				srcIndices[srcDimIdx] = 0 // Broadcast from dimension of size 1
			}
		}
		result.data[i] = t.Get(srcIndices...)
	}
	return result
}

// ExpandDims 在指定的 `axis` 位置插入一个大小为 1 的新维度。
func (t *Tensor) ExpandDims(axis int) *Tensor {
	dims := t.Ndim()
	// 允许在最后插入，所以有效范围是 [-dims-1, dims]
	if axis < 0 {
		axis += dims + 1
	}
	if axis < 0 || axis > dims {
		panic(fmt.Sprintf("invalid axis %d for expand_dims on a %d-dimensional tensor", axis, dims))
	}

	newShape := make([]int, dims+1)
	copy(newShape[:axis], t.shape[:axis])
	newShape[axis] = 1
	copy(newShape[axis+1:], t.shape[axis:])

	return t.Reshape(newShape...)
}

// =================================================================================================
// == 统计与归约 (Statistics & Reductions) ==
// =================================================================================================

// Sum 计算张量元素的和。
// 如果不提供 `axes`，则计算所有元素的和，返回一个标量张量。
// 否则，沿指定的轴进行求和。
func (t *Tensor) Sum(axes ...int) *Tensor {
	// 全局求和
	if len(axes) == 0 {
		sum := 0.0
		for _, val := range t.data {
			sum += val
		}
		return NewTensorFromData([]float64{sum}, 1)
	}

	// 沿轴求和
	return t.reduceAxes(axes, func(slice []float64) float64 {
		sum := 0.0
		for _, v := range slice {
			sum += v
		}
		return sum
	})
}

// Mean 计算张量元素的平均值。
// 如果不提供 `axes`，则计算所有元素的平均值，返回一个标量张量。
// 否则，沿指定的轴计算平均值。
func (t *Tensor) Mean(axes ...int) *Tensor {
	// 全局平均值
	if len(axes) == 0 {
		sum := 0.0
		for _, val := range t.data {
			sum += val
		}
		return NewTensorFromData([]float64{sum / float64(t.size)}, 1)
	}

	// 沿轴求平均
	return t.reduceAxes(axes, func(slice []float64) float64 {
		sum := 0.0
		for _, v := range slice {
			sum += v
		}
		return sum / float64(len(slice))
	})
}

// Variance 计算张量元素的方差。
// 如果不提供 `axes`，则计算所有元素的方差，返回一个标量张量。
// 否则，沿指定的轴计算方差。
func (t *Tensor) Variance(axes ...int) *Tensor {
	mean := t.Mean(axes...)
	var diff *Tensor

	if len(axes) == 0 {
		// Global variance
		diff = t.Sub(mean) // Broadcasting a scalar
	} else {
		// Axis-wise variance
		// We need to expand the mean tensor to match the original tensor's dimensions for subtraction.
		expandedMean := mean
		originalNdim := t.Ndim()
		meanNdim := mean.Ndim()

		// Add dimensions to the end of the mean shape until it matches the original number of dimensions
		for i := 0; i < originalNdim-meanNdim; i++ {
			expandedMean = expandedMean.ExpandDims(expandedMean.Ndim())
		}
		diff = t.Sub(expandedMean)
	}

	squaredDiff := diff.Mul(diff)
	return squaredDiff.Mean(axes...)
}

// Max 找出张量中的最大值。
// 如果不提供 `axes`，则找出所有元素中的最大值，返回一个标量张量。
// 否则，沿指定的轴找出最大值。
func (t *Tensor) Max(axes ...int) *Tensor {
	if t.size == 0 {
		panic("cannot compute max of an empty tensor")
	}
	if len(axes) == 0 {
		max := t.data[0]
		for i := 1; i < t.size; i++ {
			if t.data[i] > max {
				max = t.data[i]
			}
		}
		return NewTensorFromData([]float64{max}, 1)
	}

	return t.reduceAxes(axes, func(slice []float64) float64 {
		max := slice[0]
		for i := 1; i < len(slice); i++ {
			if slice[i] > max {
				max = slice[i]
			}
		}
		return max
	})
}

// Argmax returns the indices of the maximum values along an axis.
func (t *Tensor) Argmax(axis int) *Tensor {
	if t.size == 0 {
		panic("cannot compute argmax of an empty tensor")
	}

	// Simplified implementation for 2D tensors along axis 1
	if t.Ndim() != 2 || axis != 1 {
		panic("Argmax is only implemented for 2D tensors along axis 1")
	}

	rows := t.shape[0]
	cols := t.shape[1]
	result := NewTensor(rows)

	for i := 0; i < rows; i++ {
		maxVal := -math.MaxFloat64
		maxIdx := -1
		offset := i * t.strides[0]
		for j := 0; j < cols; j++ {
			val := t.data[offset+j]
			if val > maxVal {
				maxVal = val
				maxIdx = j
			}
		}
		result.data[i] = float64(maxIdx)
	}

	return result
}

// =================================================================================================
// == ML 相关函数 (ML-related Functions) ==
// =================================================================================================

// ReLU 激活函数 (Rectified Linear Unit)。
func (t *Tensor) ReLU() *Tensor {
	result := NewTensor(t.shape...)
	for i, val := range t.data {
		if val > 0 {
			result.data[i] = val
		} else {
			result.data[i] = 0
		}
	}
	return result
}

// Softmax 激活函数，通常用于分类任务的输出层。
// `axis` 指定应用 Softmax 的维度。为保证数值稳定性，内部会减去最大值。
func (t *Tensor) Softmax(axis int) *Tensor {
	if axis < 0 {
		axis += t.Ndim()
	}
	if axis < 0 || axis >= t.Ndim() {
		panic(fmt.Sprintf("invalid axis %d for softmax", axis))
	}

	// TODO: 实现一个更通用的 Softmax, 当前简化版仅对最后一个维度有效
	if axis != t.Ndim()-1 {
		panic("softmax is currently only implemented for the last axis")
	}

	result := t.Clone()

	// 在最后一个维度上进行操作
	chunkSize := t.shape[t.Ndim()-1]
	numChunks := t.size / chunkSize

	for i := 0; i < numChunks; i++ {
		offset := i * chunkSize
		chunk := result.data[offset : offset+chunkSize]

		// 1. 找到块中的最大值以保证数值稳定性
		maxVal := chunk[0]
		for j := 1; j < chunkSize; j++ {
			if chunk[j] > maxVal {
				maxVal = chunk[j]
			}
		}

		// 2. 计算 exp(x - max) 并求和
		sum := 0.0
		for j := 0; j < chunkSize; j++ {
			chunk[j] = math.Exp(chunk[j] - maxVal)
			sum += chunk[j]
		}

		// 3. 归一化
		for j := 0; j < chunkSize; j++ {
			chunk[j] /= sum
		}
	}

	return result
}

// LayerNorm 层归一化。
// 通常在最后一个维度上进行归一化。`gamma` 和 `beta` 是可学习的缩放和平移参数。
func (t *Tensor) LayerNorm(gamma, beta *Tensor, eps float64) *Tensor {
	lastDim := t.Ndim() - 1

	// 计算均值和方差
	mean := t.Mean(lastDim)

	// 广播 mean 以计算差值
	diff := t.Sub(mean.ExpandDims(lastDim))

	// 计算方差
	variance := diff.Mul(diff).Mean(lastDim)

	// 标准化
	std := variance.AddScalar(eps).Sqrt()
	normalized := diff.Div(std.ExpandDims(lastDim))

	// 缩放和平移
	if gamma != nil {
		expandedGamma := gamma
		for i := 0; i < normalized.Ndim()-gamma.Ndim(); i++ {
			expandedGamma = expandedGamma.ExpandDims(0)
		}
		normalized = normalized.Mul(expandedGamma)
	}
	if beta != nil {
		expandedBeta := beta
		for i := 0; i < normalized.Ndim()-beta.Ndim(); i++ {
			expandedBeta = expandedBeta.ExpandDims(0)
		}
		normalized = normalized.Add(expandedBeta)
	}

	return normalized
}

// =================================================================================================
// == 辅助函数与工具 (Helpers & Utilities) ==
// =================================================================================================

// String 实现 fmt.Stringer 接口，用于优雅地打印张量。
func (t *Tensor) String() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Tensor(shape=%v, data=\n", t.shape))

	// 递归打印
	indices := make([]int, t.Ndim())
	t.stringRecursive(&sb, indices, 0)

	sb.WriteString(")")
	return sb.String()
}

// stringRecursive 是 String() 的递归辅助函数，用于格式化多维数组的输出。
func (t *Tensor) stringRecursive(sb *strings.Builder, indices []int, depth int) {
	if depth == t.Ndim() {
		sb.WriteString(fmt.Sprintf("%.4f", t.Get(indices...)))
		return
	}

	sb.WriteString(strings.Repeat(" ", depth))
	sb.WriteString("[")
	for i := 0; i < t.shape[depth]; i++ {
		indices[depth] = i
		t.stringRecursive(sb, indices, depth+1)
		if i < t.shape[depth]-1 {
			sb.WriteString(", ")
			if depth < t.Ndim()-1 {
				sb.WriteString("\n")
			}
		}
	}
	sb.WriteString("]")
}

// Clone 创建一个张量的深拷贝。
func (t *Tensor) Clone() *Tensor {
	result := NewTensor(t.shape...)
	copy(result.data, t.data)
	if t.Grad != nil {
		result.Grad = t.Grad.Clone()
	}
	result.RequiresGrad = t.RequiresGrad
	return result
}

func (t *Tensor) sameShape(other *Tensor) bool {
	if len(t.shape) != len(other.shape) {
		return false
	}
	for i := range t.shape {
		if t.shape[i] != other.shape[i] {
			return false
		}
	}
	return true
}

// broadcastShapes 是一个简化的广播形状计算函数占位符。
// 一个完整的实现会处理更复杂的广播规则（例如，NumPy的广播语义）。
// 当前版本仅支持两个张量具有相同数量的维度，或者其中一个是一维标量张量。
// 它会 panic 如果遇到不支持的广播情况。
func broadcastShapes(a, b []int) ([]int, []int, []int) {
	// This is a simplified placeholder. A real implementation would be more complex.
	if len(a) != len(b) {
		// A real implementation would handle this by prepending dims of size 1.
		panic("Broadcasting between different numbers of dimensions is not supported in this placeholder.")
	}
	// A real implementation would also check for dimensions of size 1.
	return a, a, b
}

// Placeholder for broadcastIterator
type broadcastIterator struct {
	tensor *Tensor
	pos    int
}

func newBroadcastIterator(t *Tensor, broadcastInfo []int) *broadcastIterator {
	return &broadcastIterator{tensor: t, pos: -1}
}

func (it *broadcastIterator) Next() float64 {
	it.pos = (it.pos + 1) % it.tensor.size
	return it.tensor.data[it.pos]
}

// reduceAxes 是一个沿指定轴进行归约操作的辅助函数。
func (t *Tensor) reduceAxes(axes []int, reduceFunc func([]float64) float64) *Tensor {
	// TODO: 实现一个完整的、高效的多轴 reduce。
	// 当前实现非常简化，仅适用于特定情况，且效率不高。

	// 简化实现：仅支持单轴
	if len(axes) != 1 {
		panic("multi-axis reduction is not implemented yet")
	}

	axis := axes[0]
	if axis < 0 {
		axis += t.Ndim()
	}
	if axis < 0 || axis >= t.Ndim() {
		panic(fmt.Sprintf("invalid axis %d for reduction", axis))
	}

	// 计算输出形状
	newShape := make([]int, 0, t.Ndim()-1)
	for i, dim := range t.shape {
		if i != axis {
			newShape = append(newShape, dim)
		}
	}
	if len(newShape) == 0 {
		newShape = []int{1}
	}

	result := NewTensor(newShape...)

	// 迭代输出张量，并为每个元素计算归约值
	// 这是一种通用但效率较低的模式
	outputIndices := make([]int, len(newShape))

	for i := 0; i < result.size; i++ {
		// 从一维索引 i 计算输出张量的多维索引
		temp := i
		for j := len(newShape) - 1; j >= 0; j-- {
			outputIndices[j] = temp % newShape[j]
			temp /= newShape[j]
		}

		// 收集用于归约的值
		valuesToReduce := make([]float64, t.shape[axis])
		inputIndices := make([]int, t.Ndim())

		// 遍历归约轴
		for j := 0; j < t.shape[axis]; j++ {
			// 构建输入张量的索引
			outputIdxPtr := 0
			for k := 0; k < t.Ndim(); k++ {
				if k == axis {
					inputIndices[k] = j
				} else {
					inputIndices[k] = outputIndices[outputIdxPtr]
					outputIdxPtr++
				}
			}
			valuesToReduce[j] = t.Get(inputIndices...)
		}

		result.data[i] = reduceFunc(valuesToReduce)
	}

	return result
}
