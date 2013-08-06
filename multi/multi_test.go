package multi

import (
	"github.com/dane-unltd/linalg/clapack"
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt"
	"github.com/dane-unltd/opt/uni"
	"github.com/kortschak/cblas"
	"math"
	"math/rand"
	"testing"
	"time"
)

var cops struct {
	cblas.Blas
	clapack.Lapack
}

func TestQuadratic(t *testing.T) {
	mat.Register(cops)
	n := 100
	xStar := mat.NewVec(n)
	xStar.AddSc(1)
	A := mat.RandN(n)
	At := A.TrView()
	AtA := mat.New(n)
	AtA.Mul(At, A)

	bTmp := mat.NewVec(n)
	bTmp.Apply(A, xStar)
	b := mat.NewVec(n)
	b.Apply(At, bTmp)
	b.Scal(-2)

	c := bTmp.Nrm2Sq()

	obj, grad := opt.MakeQuadratic(AtA, b, c)

	m1 := NewModel(n, obj, grad, nil)
	m1.AddCallback(NewDisplay(n).Update)

	solver := NewSteepestDescent()
	solver.TolRel = 0

	err := solver.Solve(m1)

	t.Log(m1.ObjX, m1.Iter, err)

	solver.LineSearch = uni.NewQuadratic(false)

	m2 := NewModel(n, obj, grad, nil)
	m2.AddCallback(NewDisplay(n).Update)

	err = solver.Solve(m2)

	t.Log(m2.ObjX, m2.Iter, err)

	solver2 := NewLBFGS()
	solver2.TolRel = 0

	m3 := NewModel(n, obj, grad, nil)
	m3.AddCallback(NewDisplay(n / 10).Update)

	err = solver2.Solve(m3)

	t.Log(m3.ObjX, m3.Iter, err)

	//constrained problems (constraints described as projection)
	proj := func(x mat.Vec) {
		for i := range x {
			if x[i] < 0 {
				x[i] = 0
			}
		}
	}

	solver3 := NewProjGrad()
	solver3.TolRel = 0

	m4 := NewModel(n, obj, grad, proj)
	m4.AddCallback(NewDisplay(n).Update)

	err = solver3.Solve(m4)

	t.Log(m4.ObjX, m4.Iter, err)

	if math.Abs(m1.ObjX) > 0.01 {
		t.Log(m1.ObjX)
		t.Log(obj(xStar))
		t.Fail()
	}
	if math.Abs(m2.ObjX) > 0.01 {
		t.Log(m2.ObjX)
		t.Log(obj(xStar))
		t.Fail()
	}
	if math.Abs(m3.ObjX) > 0.01 {
		t.Log(m3.ObjX)
		t.Log(obj(xStar))
		t.Fail()
	}
	if math.Abs(m4.ObjX) > 0.01 {
		t.Log(m4.ObjX)
		t.Log(obj(xStar))
		t.Fail()
	}
}

func TestRosenbrock(t *testing.T) {
	n := 200
	var err error
	scale := 10.0
	xInit := mat.NewVec(n)
	for i := 0; i < n; i++ {
		xInit[i] = rand.Float64() * scale
	}

	obj, grad := opt.MakeRosenbrock()

	m1 := NewModel(n, obj, grad, nil)
	m1.SetX(xInit, true)
	m1.AddCallback(NewDisplay(1000).Update)
	solver := NewSteepestDescent()
	solver.TolRel = 0
	solver.IterMax = 100000

	err = solver.Solve(m1)
	t.Log(m1.ObjX, m1.Iter, err)

	solver.LineSearch = uni.NewQuadratic(false)

	m2 := NewModel(n, obj, grad, nil)
	m2.SetX(xInit, true)

	//Example on how to use a callback to display information
	//Here we could plug in something more sophisticated
	m2.AddCallback(NewDisplay(1000).Update)

	//Registering a history which stores time and objective values
	hist := History{T: make([]time.Duration, 0), Obj: make([]float64, 0)}
	m2.AddCallback(hist.Update)

	err = solver.Solve(m2)
	t.Log(m2.ObjX, m2.Iter, err)

	solver2 := NewLBFGS()
	solver2.TolRel = 0

	m3 := NewModel(n, obj, grad, nil)
	m3.SetX(xInit, true)
	m3.AddCallback(NewDisplay(10).Update)
	err = solver2.Solve(m3)
	t.Log(m3.ObjX, m3.Iter, err)

	m4 := NewModel(n, obj, grad, nil)
	m4.SetX(xInit, true)
	m4.AddCallback(NewDisplay(10).Update)
	solver2.LineSearch = uni.NewQuadratic(false)
	err = solver2.Solve(m4)
	t.Log(m4.ObjX, m4.Iter, err)

	if math.Abs(m1.ObjX) > 0.1 {
		t.Log(m1.ObjX)
		t.Fail()
	}
	if math.Abs(m2.ObjX) > 0.1 {
		t.Log(m2.ObjX)
		t.Fail()
	}
	if math.Abs(m3.ObjX) > 0.1 {
		t.Log(m3.ObjX)
		t.Fail()
	}
	if math.Abs(m4.ObjX) > 0.1 {
		t.Log(m4.ObjX)
		t.Fail()
	}
}
