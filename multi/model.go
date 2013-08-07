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

	initialGradNorm float64
	initialTime     time.Time
	gradNorm        float64

	oldX    mat.Vec
	oldObjX float64

	Params Params

	callbacks []func(m *Model) Status
}

func NewModel(n int, obj func(mat.Vec) float64, grad func(mat.Vec, mat.Vec), proj func(mat.Vec)) *Model {
	m := &Model{}
	m.N = n
	m.Obj = obj
	m.Grad = grad
	m.Proj = proj
	m.ObjX = math.NaN()
	m.callbacks = make([]func(m *Model) Status, 0)
	m.Params = Params{
		FunTolAbs: 1e-15,
		FunTolRel: 1e-15,
		XTolAbs:   1e-6,
		XTolRel:   1e-2,

		IterMax: 1000,
		TimeMax: time.Second,
	}
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

func (m *Model) AddCallback(cb func(m *Model) Status) {
	m.callbacks = append(m.callbacks, cb)
}

func (m *Model) ClearCallbacks() {
	m.callbacks = m.callbacks[0:0]
}

func (m *Model) doCallbacks() Status {
	var status Status
	for _, cb := range m.callbacks {
		st := cb(m)
		if st != 0 {
			status = st
		}
	}
	return status
}

func (m *Model) checkConvergence() Status {
	if math.Abs(m.gradNorm) < m.Params.FunTolAbs {
		return GradAbsConv
	}
	if math.Abs((m.gradNorm)/m.initialGradNorm) < m.Params.FunTolRel {
		return GradRelConv
	}
	if math.Abs(m.ObjX-m.oldObjX) < m.Params.FunTolAbs {
		return ObjAbsConv
	}
	if math.Abs((m.ObjX-m.oldObjX)/m.ObjX) < m.Params.FunTolRel {
		return ObjRelConv
	}

	if m.Iter > m.Params.IterMax {
		return IterLimit
	}
	if m.Time > m.Params.TimeMax {
		return TimeLimit
	}
	return NotTerminated
}

func (m *Model) init() {
	m.initialGradNorm = m.GradX.Nrm2()
	m.initialTime = time.Now()
	m.oldX = mat.NewVec(m.N).Scal(math.NaN())
	m.Iter = 0
}

func (m *Model) update() Status {
	m.Time = time.Since(m.initialTime)
	m.gradNorm = m.GradX.Nrm2()
	if status := m.doCallbacks(); status != 0 {
		return status
	}
	if status := m.checkConvergence(); status != 0 {
		return status
	}

	m.oldX.Copy(m.X)
	m.oldObjX = m.ObjX
	m.Iter++

	return NotTerminated
}
