package core_test

import (
	"reflect"
	"testing"

	"github.com/pplmx/algo-go/llm/transformer/core"
)

func TestAddMatrices(t *testing.T) {
	type args struct {
		a core.Matrix
		b core.Matrix
	}
	tests := []struct {
		name string
		args args
		want core.Matrix
	}{
		{
			name: "2x2 matrices",
			args: args{
				a: core.Matrix{{1, 2}, {3, 4}},
				b: core.Matrix{{5, 6}, {7, 8}},
			},
			want: core.Matrix{{6, 8}, {10, 12}},
		},
		{
			name: "1x3 matrices",
			args: args{
				a: core.Matrix{{1, 2, 3}},
				b: core.Matrix{{4, 5, 6}},
			},
			want: core.Matrix{{5, 7, 9}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := core.AddMatrices(tt.args.a, tt.args.b); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("AddMatrices() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatMul(t *testing.T) {
	type args struct {
		a core.Matrix
		b core.Matrix
	}
	tests := []struct {
		name string
		args args
		want core.Matrix
	}{
		{
			name: "2x3 and 3x2 matrices",
			args: args{
				a: core.Matrix{{1, 2, 3}, {4, 5, 6}},
				b: core.Matrix{{7, 8}, {9, 10}, {11, 12}},
			},
			want: core.Matrix{{58, 64}, {139, 154}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := core.MatMul(tt.args.a, tt.args.b); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("MatMul() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatMul_Panic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()

	a := core.Matrix{{1, 2}, {3, 4}}
	b := core.Matrix{{5, 6}, {7, 8}, {9, 10}} // Incompatible dimensions
	core.MatMul(a, b)
}

func TestTranspose(t *testing.T) {
	type args struct {
		m core.Matrix
	}
	tests := []struct {
		name string
		args args
		want core.Matrix
	}{
		{
			name: "2x3 matrix",
			args: args{
				m: core.Matrix{{1, 2, 3}, {4, 5, 6}},
			},
			want: core.Matrix{{1, 4}, {2, 5}, {3, 6}},
		},
		{
			name: "1x4 matrix",
			args: args{
				m: core.Matrix{{1, 2, 3, 4}},
			},
			want: core.Matrix{{1}, {2}, {3}, {4}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := core.Transpose(tt.args.m); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Transpose() = %v, want %v", got, tt.want)
			}
		})
	}
}
