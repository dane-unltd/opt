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

	p := NewParams()
	p.FunTolAbs = 0
	p.FunTolRel = 0

	in := NewSolution()
	in.SetX(0.5)
	in.ObjLB, in.DerivLB = fun.ValDeriv(in.LB)

	t.Log(fun, in, p)
	r := NewQuadratic().Solve(fun, in, p)

	t.Log(r.LB, r.X, r.UB, r.Status)
	t.Log(r.ObjX - fun.OptVal())
	t.Log(r.FunEvals, r.Status)

	if math.Abs(r.X-fun.OptLoc()) > 0.01 {
		t.Fail()
	}

	r = NewArmijo().Solve(fun, in, p)
	t.Log(r.FunEvals, r.Status)

	if r.ObjX >= r.ObjLB {
		t.Fail()
	}

	p.IterMax = 10
	p.Inexact = false

	r = NewCubic().Solve(fun, in, p)

	t.Log(r.LB, r.X, r.UB, r.Status)
	t.Log(r.ObjX - fun.OptVal())
	t.Log(r.FunEvals, r.Status)
	if math.Abs(r.X-fun.OptLoc()) > 0.01 {
		t.Fail()
	}
}
