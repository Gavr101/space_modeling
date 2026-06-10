# Colleagues' reference residual insights

Горизонт прогноза: 12 часов. Ошибки считаются по последней точке прогноза относительно precise-orbit reference. Baseline стоит первым; остальные строки отсортированы по среднему изменению финального положения модели относительно baseline по всем спутникам.

| Модель | Среднее влияние, м | CASSIOPE / Swarm-E \|dr\|, км | CASSIOPE / Swarm-E \|dv\|, км/с | Sentinel-1B \|dr\|, км | Sentinel-1B \|dv\|, км/с | Sentinel-1A \|dr\|, км | Sentinel-1A \|dv\|, км/с | Swarm B \|dr\|, км | Swarm B \|dv\|, км/с | Swarm C \|dr\|, км | Swarm C \|dv\|, км/с | Swarm A \|dr\|, км | Swarm A \|dv\|, км/с | Картинка |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 0.000 | 0.549855 | 0.00044602 | 0.513312 | 0.00077337 | 3.235293 | 0.00360380 | 2.204943 | 0.00214646 | 8.268206 | 0.00888696 | 8.186140 | 0.00880948 |  |
| 2. Gravity Field EGM2008 | 4653.623 | 3.993875 | 0.00361228 | 4.760690 | 0.00499608 | 4.919342 | 0.00520842 | 9.213837 | 0.00997424 | 2.622167 | 0.00285641 | 2.607370 | 0.00286890 | [2_gravity_field_egm2008.png](2_gravity_field_egm2008.png) |
| 1. rk4_fixed integrator | 507.882 | 0.848604 | 0.00056587 | 0.944558 | 0.00123242 | 2.795152 | 0.00313521 | 2.766186 | 0.00276582 | 7.682855 | 0.00823904 | 7.600818 | 0.00816162 | [1_rk4_fixed_integrator.png](1_rk4_fixed_integrator.png) |
| 3. Atmosphere NRLMSISE-00 | 215.140 | 0.772778 | 0.00051483 | 0.517675 | 0.00077800 | 3.188383 | 0.00355363 | 2.407864 | 0.00237214 | 7.925169 | 0.00850625 | 7.844205 | 0.00842995 | [3_atmosphere_nrlmsise_00.png](3_atmosphere_nrlmsise_00.png) |
| 7. Solid Earth tides | 10.344 | 0.547874 | 0.00044620 | 0.496453 | 0.00075453 | 3.250021 | 0.00361984 | 2.214374 | 0.00215693 | 8.274460 | 0.00889388 | 8.192074 | 0.00881603 | [7_solid_earth_tides.png](7_solid_earth_tides.png) |
| 8. Schwarzschild relativity | 1.179 | 0.549486 | 0.00044615 | 0.512173 | 0.00077213 | 3.236486 | 0.00360508 | 2.203660 | 0.00214505 | 8.269509 | 0.00888840 | 8.187443 | 0.00881092 | [8_schwarzschild_relativity.png](8_schwarzschild_relativity.png) |
| 4. Cylindrical Earth shadow | 0.903 | 0.551624 | 0.00044516 | 0.513312 | 0.00077337 | 3.234941 | 0.00360347 | 2.204404 | 0.00214579 | 8.268075 | 0.00888686 | 8.186168 | 0.00880955 | [4_cylindrical_earth_shadow.png](4_cylindrical_earth_shadow.png) |
| 5. Conical Earth shadow | 0.850 | 0.551624 | 0.00044516 | 0.513312 | 0.00077337 | 3.235177 | 0.00360371 | 2.204410 | 0.00214580 | 8.268173 | 0.00888696 | 8.186104 | 0.00880948 | [5_conical_earth_shadow.png](5_conical_earth_shadow.png) |
| 6. Earth IR radiation | 0.236 | 0.549734 | 0.00044605 | 0.513010 | 0.00077304 | 3.235609 | 0.00360414 | 2.204767 | 0.00214627 | 8.268384 | 0.00888716 | 8.186318 | 0.00880968 | [6_earth_ir_radiation.png](6_earth_ir_radiation.png) |

## Сохраненные картинки

- [2. Gravity Field EGM2008](2_gravity_field_egm2008.png)
- [1. rk4_fixed integrator](1_rk4_fixed_integrator.png)
- [3. Atmosphere NRLMSISE-00](3_atmosphere_nrlmsise_00.png)
- [7. Solid Earth tides](7_solid_earth_tides.png)
- [8. Schwarzschild relativity](8_schwarzschild_relativity.png)
- [4. Cylindrical Earth shadow](4_cylindrical_earth_shadow.png)
- [5. Conical Earth shadow](5_conical_earth_shadow.png)
- [6. Earth IR radiation](6_earth_ir_radiation.png)

## Machine-readable results

- [results_per_satellite.csv](results_per_satellite.csv)
- [results_per_satellite.json](results_per_satellite.json)
