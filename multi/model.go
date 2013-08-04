package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"math"
	"time"
)

type Model struct {
	Obj     func(x mat.Vec) float64
	Grad    func(x, g mat.Vec)
	Hessian func(x mat.Vec, H *mat.Dense)
	Proj    func(x mat.Vec)

	X        mat.Vec
	ObjX     float64
	GradX    mat.Vec
	HessianX *mat.Dense

	Iter int
	Time time.Duration

	callback func(m *Model)
}

type History struct {
	T    []time.Duration
	X    []mat.Vec
	Obj  []float64
	Grad []mat.Vec
}

func NewModel(obj func(mat.Vec) float64, grad func(mat.Vec, mat.Vec), x mat.Vec) *Model {
	m := &Model{}
	m.Obj = obj
	m.Grad = grad
	m.X = x
	m.ObjX = math.NaN()
	return m
}

func (h *History) Update(m *Model) {
	if h.T != nil {
		h.T = append(h.T, m.Time)
	}
	if h.X != nil {
		xt := make(mat.Vec, len(m.X))
		xt.Copy(m.X)
		h.X = append(h.X, xt)
	}
	if h.Grad != nil {
		if m.GradX != nil {
			gt := make(mat.Vec, len(m.GradX))
			gt.Copy(m.GradX)
			h.Grad = append(h.Grad, gt)
		}
	}
	if h.Obj != nil {
		h.Obj = append(h.Obj, m.ObjX)
	}
}
