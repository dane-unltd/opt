package uni

import (
	"errors"
	"math"
	"time"
)

//Inexact line search using Armijo's rule.
type Armijo struct {
	IterMax int
}

func NewArmijo() *Armijo {
	return &Armijo{IterMax: 100}
}

func (s *Armijo) Solve(m *Model) error {
	var err error = nil

	startT := time.Now()

	beta := 0.5

	step := m.x - m.lb
	maxStep := m.ub - m.lb

	if math.IsNaN(m.derivLB) {
		m.derivLB = m.deriv(m.lb)
	}
	if math.IsNaN(m.objLB) {
		m.objLB = m.obj(m.lb)
	}
	if math.IsNaN(step) || step <= 0 || step > maxStep {
		if math.IsInf(m.ub, 1) {
			step = 1
		} else {
			step = maxStep
		}
	}

	if m.derivLB > 0 {
		err = errors.New("Armijo: No progress possible")
		return err
	}

	m.x = m.lb + step
	m.objX = m.obj(m.x)

	m.iter = 0

	if m.objX-m.objLB > 0.5*m.derivLB*step {
		fPrev := m.objX
		step *= beta
		for {
			m.iter++
			if m.iter == s.IterMax {
				break
			}
			m.x = m.lb + step
			m.objX = m.obj(m.x)
			m.time = time.Since(startT)
			m.DoCallbacks()

			if m.objX-m.objLB <= 0.5*m.derivLB*step {
				if fPrev < m.objX {
					step /= beta
					m.x = m.lb + step
					m.objX = fPrev
				}
				break
			}
			fPrev = m.objX
			step *= beta
		}
	} else {
		fPrev := m.objX
		if step == maxStep {
			goto done
		}
		step /= beta
		if step > maxStep {
			step = maxStep
		}
		for {
			m.iter++
			if m.iter == s.IterMax {
				break
			}
			m.x = m.lb + step
			m.objX = m.obj(m.x)
			m.time = time.Since(startT)
			m.DoCallbacks()

			if m.objX-m.objLB > 0.5*m.derivLB*step {
				if fPrev < m.objX {
					step *= beta
					m.x = m.lb + step
					m.objX = fPrev
				}
				break
			}
			fPrev = m.objX
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
	if m.iter == s.IterMax {
		err = errors.New("Maximum number of Iterations reached")
	}

	return err
}
