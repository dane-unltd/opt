package uni

import (
	"math"
)

//Inexact line search using Armijo's rule.
type Armijo struct {
}

func NewArmijo() *Armijo {
	return &Armijo{}
}

func (s *Armijo) Solve(m *Model) {
	m.init(false, false)

	beta := 0.5
	sigma := 0.2

	step := m.X - m.LB
	maxStep := m.UB - m.LB

	if math.IsNaN(m.DerivLB) {
		panic("have to set derivation of lower bound for Armijo")
	}
	if math.IsNaN(m.ObjLB) {
		m.ObjLB = m.Obj.Val(m.LB)
	}

	if m.DerivLB > 0 {
		m.Status = Fail
		return
	}

	m.X = m.LB + step
	m.ObjX = m.Obj.Val(m.X)

	if m.ObjX-m.ObjLB > sigma*m.DerivLB*step {
		fPrev := m.ObjX
		step *= beta
		for {
			m.X = m.LB + step
			m.ObjX = m.Obj.Val(m.X)

			if m.Status = m.update(); m.Status != 0 {
				break
			}

			if m.ObjX-m.ObjLB <= sigma*m.DerivLB*step {
				if fPrev < m.ObjX {
					step /= beta
					m.X = m.LB + step
					m.ObjX = fPrev
				}
				break
			}
			fPrev = m.ObjX
			step *= beta
		}
	} else {
		fPrev := m.ObjX
		if step == maxStep {
			return
		}
		step /= beta
		if step > maxStep {
			step = maxStep
		}
		for {
			m.X = m.LB + step
			m.ObjX = m.Obj.Val(m.X)
			if m.Status = m.update(); m.Status != 0 {
				break
			}

			if m.ObjX-m.ObjLB > sigma*m.DerivLB*step {
				if fPrev < m.ObjX {
					step *= beta
					m.X = m.LB + step
					m.ObjX = fPrev
				}
				break
			}
			fPrev = m.ObjX
			if step == maxStep {
				return
			}
			step /= beta
			if step > maxStep {
				step = maxStep
			}
		}
	}
}
