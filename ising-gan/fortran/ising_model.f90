PROGRAM IsingModel
  IMPLICIT NONE

  INTEGER, PARAMETER :: L = 128                       ! Lattice size (LxL grid)
  INTEGER, PARAMETER :: equilibrium_cycle = 500000    ! Monte Carlo cycles (1 MC cycle = L*L MC steps or trial moves) for equilibration
  INTEGER, PARAMETER :: measurement_cycle = 1000      ! Monte Carlo cycles (1 MC cycle = L*L MC steps or trial moves) for production/measurement
  INTEGER, PARAMETER :: measurement_cycle_gap = 100   ! Monte Carlo cycles to skip for measurements
  !REAL, PARAMETER :: T = 2.0           ! Temperature
  REAL, PARAMETER :: J = 1.0            ! Interaction constant

  CHARACTER :: phase
  INTEGER :: row, col, step, istep
  ! INTEGER :: spins(L, L)
  INTEGER(KIND=1) :: spins(L, L)
  INTEGER :: x, y, deltaE, iT, N
  INTEGER, PARAMETER :: rk = SELECTED_REAL_KIND(p=15, r=307)
  REAL(KIND=16) :: T,dT
  REAL :: beta, rand_num
  REAL :: energy, magnetization, correlation_function, correlation_length
  REAL :: energy2, magnetization2, specific_heat, susceptibility
  REAL :: total_energy, total_energy2, total_magnetization, total_magnetization2
  REAL, DIMENSION(L) :: correlation
  CHARACTER (len=256) :: filename

  CALL RANDOM_SEED()

  ! Construct the filename with parameters
  write(filename, '(A,I3.3,A,I7.7,A)') 'Ising_2D_L_', L, '_cycles_', measurement_cycle, '.dat'
    
  ! Open file for writing, unit number 10
  open(unit=10, file=trim(filename), status='replace', action='write')
    
  ! Write header
  write(10,*) '# T, total_energy/N, total_magnetization/N, specific_heat,' // &
               'susceptibility, correlation_function, correlation_length'

  N  = L * L
  T  = 4.0
  dT = 0.05
  dT = (4.0-1.0)/49.0

  write(filename, '(A,I3.3,A,I7.7,A)') 'Ising_2D_L_', L, '_cycles_', measurement_cycle, '.csv'
  open(unit=20, file=trim(filename), status='replace', action='write')
  
  WRITE(20, '(A,A)', ADVANCE='no') "Temperature,", "Phase,"
  DO row = 1, L
    DO col = 1, L
      WRITE(20, '(A,I0,A)', ADVANCE='no') "spin_", (row-1)*L + col-1, ","
    END DO
  END DO
  WRITE(20, '(I2)', ADVANCE='yes')

  DO iT = 1, 50

    beta = 1.0 / T

    CALL InitializeSpins(spins)

    DO step = 1, equilibrium_cycle
       CALL PerformCycle(spins, L, beta, J)
    END DO

    total_energy  = 0.0
    total_energy2 = 0.0
    total_magnetization  = 0.0
    total_magnetization2 = 0.0
    correlation = 0.0

    write(filename, '(A,I3.3,A,I7.7,A,I3.3,A)') 'Ising_2D_L_', L, '_cycles_', measurement_cycle, '_T_', iT, '.csv'
    !write(filename, '(A,I3.3,A,I7.7,A,F5.3,A)') 'Ising_2D_L_', L, '_cycles_', measurement_cycle, '_T_', T, '.csv'
    open(unit=30, file=trim(filename), status='replace', action='write')
    
    WRITE(30, '(A,A)', ADVANCE='no') "Temperature,", "Phase,"
    DO row = 1, L
      DO col = 1, L
        WRITE(30, '(A,I0,A)', ADVANCE='no') "spin_", (row-1)*L + col-1, ","
      END DO
    END DO
    WRITE(30, '(I2)', ADVANCE='yes')

    write(filename, '(A,I3.3,A,I7.7,A,I3.3,A)') 'Ising_2D_L_', L, '_cycles_', measurement_cycle, '_T_', iT, '.bin'
    !write(filename, '(A,I3.3,A,I7.7,A,F5.3,A)') 'Ising_2D_L_', L, '_cycles_', measurement_cycle, '_T_', T, '.bin'
    open(unit=40, file=trim(filename), status='replace', action='write',form='unformatted',access='stream')
    
    DO step = 1, measurement_cycle

      DO istep = 1, measurement_cycle_gap
        CALL PerformCycle(spins, L, beta, J)
      END DO

      energy = TotalEnergy(spins, L, J)
      magnetization = ABS(TotalMagnetization(spins, L))
      CALL CalculateCorrelation(spins, L, correlation)

      total_energy  = total_energy  + energy
      total_energy2 = total_energy2 + energy**2
      total_magnetization  = total_magnetization  + magnetization
      total_magnetization2 = total_magnetization2 + magnetization**2
      
      IF(T<2.269) THEN
         phase='F'
      ELSE 
         phase='P'
      ENDIF
      WRITE(20, '(F17.14,A,A,A)', ADVANCE='no') T,',',phase,','
      DO row = 1, L
         DO col = 1, L
            WRITE(20, '(I2,A)', ADVANCE='no') spins(row, col),','
         END DO
      END DO
      WRITE(20, '(I2)', ADVANCE='yes')

      WRITE(30, '(F17.14,A,A,A)', ADVANCE='no') T,',',phase,','
      DO row = 1, L
         DO col = 1, L
            WRITE(30, '(I2,A)', ADVANCE='no') spins(row, col),','
         END DO
      END DO
      WRITE(30, '(I2)', ADVANCE='yes')
            
      WRITE(40) spins
    END DO
    close(40)
    close(30)
    
    total_energy  = total_energy  / measurement_cycle
    total_energy2 = total_energy2 / measurement_cycle
    total_magnetization  = total_magnetization  / measurement_cycle
    total_magnetization2 = total_magnetization2 / measurement_cycle

    specific_heat  = (total_energy2        - total_energy**2       ) / (T**2 * N)
    susceptibility = (total_magnetization2 - total_magnetization**2) / (T    * N)
    correlation_function = SUM(correlation) / (L * measurement_cycle)
    correlation_length = -1.0 / LOG(correlation_function)

    WRITE(10, '(F5.3, E14.6, E14.6, E14.6, E14.6, E14.6, E14.6)') T, &
               total_energy/N, total_magnetization/N, specific_heat, &
               susceptibility, correlation_function, correlation_length
               
    T = T - dT
  END DO
  ! Close the file
  close(20)
  close(10)
  
CONTAINS

  SUBROUTINE InitializeSpins(spins)
    INTEGER(KIND=1), INTENT(OUT) :: spins(L, L)
    INTEGER :: row, col
    REAL :: rand_num

    DO row = 1, L
       DO col = 1, L
          CALL RANDOM_NUMBER(rand_num)
          IF (rand_num < 0.5) THEN
             spins(row, col) = 1
          ELSE
             spins(row, col) = -1
          END IF
       END DO
    END DO
  END SUBROUTINE InitializeSpins

  SUBROUTINE PerformCycle(spins, L, beta, J)
    INTEGER(KIND=1), INTENT(INOUT) :: spins(L, L)
    INTEGER, INTENT(IN) :: L
    REAL, INTENT(IN) :: beta, J
    INTEGER :: x, y, N, step
    REAL :: rand_num,deltaE
    
    N = L*L
    DO step = 1, N
      CALL RandomPosition(L, x, y)
      deltaE = 2 * J * spins(x, y) * NeighborSum(spins, x, y, L)

      IF (deltaE <= 0) THEN
         spins(x, y) = -spins(x, y)
      ELSE
         CALL RANDOM_NUMBER(rand_num)
         IF (rand_num < EXP(-beta * deltaE)) THEN
            spins(x, y) = -spins(x, y)
         END IF
      END IF
    END DO
  END SUBROUTINE PerformCycle

  SUBROUTINE RandomPosition(L, x, y)
    INTEGER, INTENT(IN) :: L
    INTEGER, INTENT(OUT) :: x, y
    REAL :: rand_num

    CALL RANDOM_NUMBER(rand_num)
    x = 1 + INT(rand_num * L)

    CALL RANDOM_NUMBER(rand_num)
    y = 1 + INT(rand_num * L)
  END SUBROUTINE RandomPosition

  INTEGER FUNCTION NeighborSum(spins, x, y, L)
    INTEGER(KIND=1), INTENT(IN) :: spins(L, L)
    INTEGER, INTENT(IN) :: x, y, L
    INTEGER :: xm, xp, ym, yp

    xm = MOD(x - 2 + L, L) + 1
    xp = MOD(x, L) + 1
    ym = MOD(y - 2 + L, L) + 1
    yp = MOD(y, L) + 1

    NeighborSum = spins(xm, y) + spins(xp, y) + spins(x, ym) + spins(x, yp)
  END FUNCTION NeighborSum

  REAL FUNCTION TotalEnergy(spins, L, J)
    INTEGER(KIND=1), INTENT(IN) :: spins(L, L)
    INTEGER, INTENT(IN) :: L
    REAL, INTENT(IN) :: J
    INTEGER :: row, col

    TotalEnergy = 0.0
    DO row = 1, L
       DO col = 1, L
          TotalEnergy = TotalEnergy - J * spins(row, col) * NeighborSum(spins, row, col, L)
       END DO
    END DO
    TotalEnergy = 0.5 * TotalEnergy  
  END FUNCTION TotalEnergy

  REAL FUNCTION TotalMagnetization(spins, L)
    INTEGER(KIND=1), INTENT(IN) :: spins(L, L)
    INTEGER, INTENT(IN) :: L
    INTEGER :: row, col

    TotalMagnetization = 0.0
    DO row = 1, L
       DO col = 1, L
          TotalMagnetization = TotalMagnetization + spins(row, col)
       END DO
    END DO
  END FUNCTION TotalMagnetization

  SUBROUTINE CalculateCorrelation(spins, L, correlation)
    INTEGER(KIND=1), INTENT(IN) :: spins(L, L)
    INTEGER, INTENT(IN) :: L
    REAL, INTENT(INOUT) :: correlation(L)
    INTEGER :: row, col, shift

    DO shift = 1, L
       correlation(shift) = 0.0
       DO row = 1, L
          DO col = 1, L
             correlation(shift) = correlation(shift) + spins(row, col) * spins(MOD(row + shift - 1, L) + 1, col)
          END DO
       END DO
       correlation(shift) = correlation(shift) / (L * L)
    END DO
  END SUBROUTINE CalculateCorrelation

END PROGRAM
