package uni

import (
	"math"
	"testing"
)

type SumExpStruct struct{}

func (s SumExpStruct) Val(x float64) (f float64) {

	// http://www.wolframalpha.com/input/?i=0.3+*+exp%28+-+3+%28x-1%29%29+%2B+exp%28x-1%29
	c1 := 0.3
	c2 := 3.0
	f = c1*math.Exp(-c2*(x-1)) + math.Exp((x - 1))
	return f
}

func (s SumExpStruct) ValDeriv(x float64) (f, d float64) {

	// http://www.wolframalpha.com/input/?i=0.3+*+exp%28+-+3+%28x-1%29%29+%2B+exp%28x-1%29
	c1 := 0.3
	c2 := 3.0
	f = c1*math.Exp(-c2*(x-1)) + math.Exp((x - 1))
	d = -c1*c2*math.Exp(-c2*(x-1)) + math.Exp((x - 1))
	return f, d
}

func (s SumExpStruct) Name() string {
	return "SumExp"
}

func (s SumExpStruct) OptVal() float64 {
	return 1.298671661900395685896941595
}

func (s SumExpStruct) OptLoc() float64 {
	return 0.9736598710855434246931247548
}

func TestUni(t *testing.T) {
	fun := SumExpStruct{}
	mdl := NewModel(fun)
	mdl.SetX(0.5)
	mdl.Params.FunTolAbs = 0
	mdl.Params.FunTolRel = 0
	NewQuadratic(false).Solve(mdl)

	t.Log(mdl.LB, mdl.X, mdl.UB, mdl.Status)
	t.Log(mdl.ObjX - fun.OptVal())
	t.Log(mdl.Iter)

	if math.Abs(mdl.X-fun.OptLoc()) > 0.01 {
		t.Fail()
	}

	mdl = NewModel(fun)
	mdl.ObjLB, mdl.DerivLB = fun.ValDeriv(mdl.LB)
	NewArmijo().Solve(mdl)
	t.Log(mdl.X)

	if mdl.ObjX >= mdl.ObjLB {
		t.Fail()
	}
}
