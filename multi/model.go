package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"math"
	"time"
)

//Describes a multi-variate optimization problem.
//The solvers in this package place the results in the different fields.
type Model struct {
	N int

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

	callbacks []func(m *Model)
}

func NewModel(n int, obj func(mat.Vec) float64, grad func(mat.Vec, mat.Vec), proj func(mat.Vec)) *Model {
	m := &Model{}
	m.N = n
	m.Obj = obj
	m.Grad = grad
	m.Proj = proj
	m.ObjX = math.NaN()
	m.callbacks = make([]func(m *Model), 0)
	return m
}

func (m *Model) SetX(x mat.Vec, cpy bool) {
	if cpy {
		if m.X == nil {
			m.X = make(mat.Vec, m.N)
		}
		m.X.Copy(x)
	} else {
		m.X = x
	}
	m.ObjX = math.NaN()
	m.GradX = nil
	m.HessianX = nil
}

func (m *Model) ChangeFun(obj func(mat.Vec) float64, grad func(mat.Vec, mat.Vec), proj func(mat.Vec)) {
	m.Obj = obj
	m.Grad = grad
	m.Proj = proj

	m.ObjX = math.NaN()
	m.GradX = nil
	m.HessianX = nil
}

func (m *Model) AddVar(x float64) {
	m.N++
	m.X = append(m.X, x)

	m.ObjX = math.NaN()
	m.GradX = nil
	m.HessianX = nil
}

func (m *Model) AddCallback(cb func(m *Model)) {
	m.callbacks = append(m.callbacks, cb)
}

func (m *Model) DoCallbacks() {
	for _, cb := range m.callbacks {
		cb(m)
	}
}

func (m *Model) ClearCallbacks() {
	m.callbacks = m.callbacks[0:0]
}
