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

func (s *Armijo) Solve(m *Model) Status {
	var status Status

	beta := 0.5
	sigma := 0.2

	step := m.X - m.LB
	maxStep := m.UB - m.LB

	if math.IsNaN(m.DerivLB) {
		m.DerivLB = m.Deriv(m.LB)
	}
	if math.IsNaN(m.ObjLB) {
		m.ObjLB = m.Obj(m.LB)
	}
	if math.IsNaN(step) || step <= 0 || step > maxStep {
		if math.IsInf(m.UB, 1) {
			step = 1
		} else {
			step = maxStep
		}
	}

	if m.DerivLB > 0 {
		status = Fail
		return status
	}

	m.X = m.LB + step
	m.ObjX = m.Obj(m.X)

	m.init()

	if m.ObjX-m.ObjLB > sigma*m.DerivLB*step {
		fPrev := m.ObjX
		step *= beta
		for {
			m.X = m.LB + step
			m.ObjX = m.Obj(m.X)

			if status = m.update(); status != 0 {
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
			goto done
		}
		step /= beta
		if step > maxStep {
			step = maxStep
		}
		for {
			m.X = m.LB + step
			m.ObjX = m.Obj(m.X)
			if status = m.update(); status != 0 {
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
				goto done
			}
			step /= beta
			if step > maxStep {
				step = maxStep
			}
		}
	}

done:

	return status
}
