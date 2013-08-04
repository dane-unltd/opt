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
)

var cops struct {
	cblas.Blas
	clapack.Lapack
}

func TestQuadratic(t *testing.T) {
	mat.Register(cops)
	n := 5
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

	x1 := mat.NewVec(n)

	m1 := NewModel(obj, grad, x1)
	solver := NewSteepestDescent()

	err := solver.Solve(m1)
	t.Log(m1.ObjX, m1.Iter, err)

	solver.LineSearch = uni.NewQuadratic()

	x2 := mat.NewVec(n)

	m2 := NewModel(obj, grad, x2)
	err = solver.Solve(m2)
	t.Log(m2.ObjX, m2.Iter, err)

	solver2 := NewLBFGS()

	x3 := mat.NewVec(n)
	m3 := NewModel(obj, grad, x3)
	err = solver2.Solve(m3)
	t.Log(m3.ObjX, m3.Iter, err)

	//constrained problems
	proj := func(x mat.Vec) {
		for i := range x {
			if x[i] < 0 {
				x[i] = 0
			}
		}
	}
	solver3 := NewProjGrad()
	x4 := mat.NewVec(n)
	m4 := NewModel(obj, grad, x4)
	m4.Proj = proj
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
	/*if math.Abs(m4.ObjX) > 0.01 {
		t.Log(m4.ObjX)
		t.Log(obj(xStar))
		t.Fail()
	}*/
}

func TestRosenbrock(t *testing.T) {
	n := 6
	var err error
	scale := 10.0
	xInit := mat.NewVec(n)
	for i := 0; i < n; i++ {
		xInit[i] = rand.Float64() * scale
	}

	obj, grad := opt.MakeRosenbrock()

	x1 := mat.NewVec(n)
	x1.Copy(xInit)

	m1 := NewModel(obj, grad, x1)
	solver := NewSteepestDescent()
	solver.TolRel = 0
	solver.IterMax = 100000

	err = solver.Solve(m1)
	t.Log(m1.ObjX, m1.Iter, err)

	solver.LineSearch = uni.NewQuadratic()

	x2 := mat.NewVec(n)
	x2.Copy(xInit)

	m2 := NewModel(obj, grad, x2)

	m2.callback = func(m *Model) {
		//	t.Log(obj)
	}

	err = solver.Solve(m2)
	t.Log(m2.ObjX, m2.Iter, err)

	solver2 := NewLBFGS()

	x3 := mat.NewVec(n)
	x3.Copy(xInit)

	m3 := NewModel(obj, grad, x3)
	err = solver2.Solve(m3)
	_ = solver2
	t.Log(m3.ObjX, m3.Iter, err)

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
}
