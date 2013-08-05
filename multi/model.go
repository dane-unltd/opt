package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"math"
	"time"
)

type Model struct {
	n int

	obj     func(x mat.Vec) float64
	grad    func(x, g mat.Vec)
	hessian func(x mat.Vec, H *mat.Dense)
	proj    func(x mat.Vec)

	x        mat.Vec
	objX     float64
	gradX    mat.Vec
	hessianX *mat.Dense

	iter int
	time time.Duration

	callbacks []func(m *Model)
}

func NewModel(n int, obj func(mat.Vec) float64, grad func(mat.Vec, mat.Vec), proj func(mat.Vec)) *Model {
	m := &Model{}
	m.n = n
	m.obj = obj
	m.grad = grad
	m.proj = proj
	m.objX = math.NaN()
	m.callbacks = make([]func(m *Model), 0)
	return m
}

func (m *Model) SetX(x mat.Vec, cpy bool) {
	if cpy {
		if m.x == nil {
			m.x = make(mat.Vec, m.n)
		}
		m.x.Copy(x)
	} else {
		m.x = x
	}
	m.objX = math.NaN()
	m.gradX = nil
	m.hessianX = nil
}

func (m *Model) ChangeFun(obj func(mat.Vec) float64, grad func(mat.Vec, mat.Vec), proj func(mat.Vec)) {
	m.obj = obj
	m.grad = grad
	m.proj = proj

	m.objX = math.NaN()
	m.gradX = nil
	m.hessianX = nil
}

func (m *Model) AddVar(x float64) {
	m.n++
	m.x = append(m.x, x)

	m.objX = math.NaN()
	m.gradX = nil
	m.hessianX = nil
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

func (m *Model) GradX() mat.Vec {
	return m.gradX
}

func (m *Model) X() mat.Vec {
	return m.x
}

func (m *Model) ObjX() float64 {
	return m.objX
}

func (m *Model) Iter() int {
	return m.iter
}

func (m *Model) Time() time.Duration {
	return m.time
}
