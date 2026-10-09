# OpenAQ Audit — Chennai

- Run at (UTC): 2026-10-09T07:18:42.273231+00:00
- Audit window: 2025-01-01 00:00 to 2026-10-08 23:00 IST (hour starts; 15504 expected hours)
- Source: OpenAQ v3 `/locations` (bbox), `/locations/{id}/sensors`, `/sensors/{id}/hours`.
- Hourly values are OpenAQ hourly aggregates of the provider's sub-hourly measurements; coverage below is computed from downloaded records, not from sensor metadata.
- HTTP requests made this run: 1985 (failed after retries: 0); sensor-months that could not be downloaded: 0.

## Locations

| id | name | provider | lat | lon | point-in-polygon zone | metadata last (UTC) |
|---|---|---|---|---|---|---|
| 378 | Alandur Bus Depot | CPCB | 12.9971 | 80.1915 | outside GCC (0.09 km from Alandur) | 2018-02-22T03:45:00Z |
| 2461 | US Diplomatic Post: Chennai | StateAir Chennai | 13.0524 | 80.2519 | 9 Teynampet | 2016-11-09T16:30:00Z |
| 2549 | IIT | CPCB | 12.9925 | 80.2374 | 13 Adyar | 2018-02-22T03:45:00Z |
| 2586 | Manali, Chennai - CPCB | CPCB | 13.1645 | 80.2629 | 2 Manali | 2026-10-07T14:30:00Z |
| 5558 | Alandur Bus Depot, Chennai - CPCB | caaqm | 12.9099 | 80.1077 | outside GCC (9.64 km from Perungudi) | 2022-10-31T01:45:00Z |
| 5655 | Velachery Res. Area, Chennai - CPCB | CPCB | 13.0052 | 80.2398 | 13 Adyar | 2026-10-07T14:30:00Z |
| 8558 | Chennai | AirNow | 13.0878 | 80.2785 | 5 Royapuram | 2026-10-09T06:30:00Z |
| 10780 | Manali Village, Chennai - TNPCB | CPCB | 13.1662 | 80.2584 | 2 Manali | 2026-10-07T14:30:00Z |
| 11578 | Royapuram, Chennai - TNPCB | CPCB | 13.1036 | 80.2909 | 5 Royapuram | 2026-10-07T14:30:00Z |
| 11579 | Kodungaiyur, Chennai - TNPCB | CPCB | 13.1278 | 80.2642 | 4 Tondiarpet | 2026-10-07T14:30:00Z |
| 11581 | Arumbakkam, Chennai - TNPCB | CPCB | 13.0664 | 80.2112 | 8 Anna Nagar | 2026-10-07T14:30:00Z |
| 12046 | Perungudi, Chennai - TNPCB | CPCB | 12.9533 | 80.2357 | 15 Sholinganallur | 2026-09-28T18:45:00Z |
| 3409319 | Alandur Bus Depot, Chennai - CPCB | CPCB | 12.9099 | 80.1077 | outside GCC (9.64 km from Perungudi) | 2025-12-02T11:00:00Z |

## Sensor coverage in the audit window (sensors with ≥1 valid hour)

| location | parameter | sensor | units | valid hours | first valid (IST) | last valid (IST) | full-period % | active-period % | dup hours | neg | sentinel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Alandur Bus Depot, Chennai - CPCB | co | 12235544 | ppb | 5702 | 2025-02-19T01:00 | 2025-12-02T16:00 | 36.78 | 82.88 | 0 | 0 | 0 |
| Alandur Bus Depot, Chennai - CPCB | no | 12235545 | ppb | 5702 | 2025-02-19T01:00 | 2025-12-02T16:00 | 36.78 | 82.88 | 0 | 0 | 0 |
| Alandur Bus Depot, Chennai - CPCB | no2 | 12235546 | ppb | 5702 | 2025-02-19T01:00 | 2025-12-02T16:00 | 36.78 | 82.88 | 0 | 0 | 0 |
| Alandur Bus Depot, Chennai - CPCB | nox | 14341857 | ppb | 5702 | 2025-02-19T01:00 | 2025-12-02T16:00 | 36.78 | 82.88 | 0 | 0 | 0 |
| Alandur Bus Depot, Chennai - CPCB | o3 | 12235547 | µg/m³ | 5702 | 2025-02-19T01:00 | 2025-12-02T16:00 | 36.78 | 82.88 | 0 | 0 | 0 |
| Alandur Bus Depot, Chennai - CPCB | pm10 | 12235548 | µg/m³ | 5701 | 2025-02-19T01:00 | 2025-12-02T16:00 | 36.77 | 82.86 | 0 | 0 | 0 |
| Alandur Bus Depot, Chennai - CPCB | pm25 | 12235549 | µg/m³ | 5702 | 2025-02-19T01:00 | 2025-12-02T16:00 | 36.78 | 82.88 | 0 | 0 | 0 |
| Alandur Bus Depot, Chennai - CPCB | relativehumidity | 14484666 | % | 781 | 2025-10-27T19:00 | 2025-12-02T16:00 | 5.04 | 90.6 | 0 | 0 | 0 |
| Alandur Bus Depot, Chennai - CPCB | so2 | 12235550 | ppb | 5702 | 2025-02-19T01:00 | 2025-12-02T16:00 | 36.78 | 82.88 | 0 | 0 | 0 |
| Alandur Bus Depot, Chennai - CPCB | temperature | 14018152 | c | 435 | 2025-09-02T12:00 | 2025-09-22T15:00 | 2.81 | 89.88 | 0 | 0 | 0 |
| Alandur Bus Depot, Chennai - CPCB | wind_direction | 14341858 | deg | 5702 | 2025-02-19T01:00 | 2025-12-02T16:00 | 36.78 | 82.88 | 0 | 0 | 0 |
| Alandur Bus Depot, Chennai - CPCB | wind_speed | 14341859 | m/s | 5702 | 2025-02-19T01:00 | 2025-12-02T16:00 | 36.78 | 82.88 | 0 | 0 | 0 |
| Arumbakkam, Chennai - TNPCB | no | 12236271 | ppb | 10719 | 2025-02-19T01:00 | 2026-10-07T19:00 | 69.14 | 74.96 | 0 | 0 | 0 |
| Arumbakkam, Chennai - TNPCB | no2 | 12236272 | ppb | 10722 | 2025-02-19T01:00 | 2026-10-07T19:00 | 69.16 | 74.98 | 0 | 0 | 0 |
| Arumbakkam, Chennai - TNPCB | nox | 14340986 | ppb | 10724 | 2025-02-19T01:00 | 2026-10-07T19:00 | 69.17 | 75.0 | 0 | 0 | 0 |
| Arumbakkam, Chennai - TNPCB | pm10 | 12236273 | µg/m³ | 10617 | 2025-02-19T01:00 | 2026-10-07T19:00 | 68.48 | 74.25 | 0 | 0 | 0 |
| Arumbakkam, Chennai - TNPCB | pm25 | 12236274 | µg/m³ | 10291 | 2025-02-19T01:00 | 2026-10-07T19:00 | 66.38 | 71.97 | 0 | 0 | 0 |
| Arumbakkam, Chennai - TNPCB | relativehumidity | 12236275 | % | 10716 | 2025-02-19T01:00 | 2026-10-07T19:00 | 69.12 | 74.94 | 0 | 0 | 0 |
| Arumbakkam, Chennai - TNPCB | so2 | 12236276 | ppb | 10723 | 2025-02-19T01:00 | 2026-10-07T19:00 | 69.16 | 74.99 | 0 | 0 | 0 |
| Arumbakkam, Chennai - TNPCB | temperature | 12236277 | c | 9636 | 2025-02-19T01:00 | 2026-08-06T16:00 | 62.15 | 75.23 | 0 | 0 | 0 |
| Arumbakkam, Chennai - TNPCB | wind_direction | 14340987 | deg | 9112 | 2025-02-25T19:00 | 2026-10-07T19:00 | 58.77 | 64.45 | 0 | 0 | 0 |
| Arumbakkam, Chennai - TNPCB | wind_speed | 14340988 | m/s | 10625 | 2025-02-19T01:00 | 2026-10-07T19:00 | 68.53 | 74.31 | 0 | 0 | 0 |
| Chennai | pm25 | 5077812 | µg/m³ | 7949 | 2025-01-01T00:00 | 2026-10-08T23:00 | 51.27 | 51.27 | 0 | 20 | 0 |
| Kodungaiyur, Chennai - TNPCB | co | 12236303 | ppb | 11601 | 2025-02-19T01:00 | 2026-10-07T19:00 | 74.83 | 81.13 | 0 | 0 | 0 |
| Kodungaiyur, Chennai - TNPCB | no | 12236304 | ppb | 12125 | 2025-02-19T01:00 | 2026-10-07T19:00 | 78.21 | 84.8 | 0 | 0 | 0 |
| Kodungaiyur, Chennai - TNPCB | no2 | 12236305 | ppb | 12101 | 2025-02-19T01:00 | 2026-10-07T19:00 | 78.05 | 84.63 | 0 | 0 | 0 |
| Kodungaiyur, Chennai - TNPCB | nox | 14340998 | ppb | 12120 | 2025-02-19T01:00 | 2026-10-07T19:00 | 78.17 | 84.76 | 0 | 0 | 0 |
| Kodungaiyur, Chennai - TNPCB | o3 | 12236306 | µg/m³ | 12084 | 2025-02-19T01:00 | 2026-10-07T19:00 | 77.94 | 84.51 | 0 | 0 | 0 |
| Kodungaiyur, Chennai - TNPCB | pm10 | 12236307 | µg/m³ | 12016 | 2025-02-19T01:00 | 2026-10-07T19:00 | 77.5 | 84.03 | 0 | 0 | 0 |
| Kodungaiyur, Chennai - TNPCB | pm25 | 12236308 | µg/m³ | 11911 | 2025-02-19T01:00 | 2026-10-07T19:00 | 76.83 | 83.3 | 0 | 0 | 0 |
| Kodungaiyur, Chennai - TNPCB | relativehumidity | 12236309 | % | 12115 | 2025-02-19T01:00 | 2026-10-07T19:00 | 78.14 | 84.73 | 0 | 0 | 0 |
| Kodungaiyur, Chennai - TNPCB | so2 | 12236310 | ppb | 11821 | 2025-02-19T01:00 | 2026-10-07T19:00 | 76.24 | 82.67 | 0 | 0 | 0 |
| Kodungaiyur, Chennai - TNPCB | temperature | 12236311 | c | 11061 | 2025-02-19T01:00 | 2026-09-06T21:00 | 71.34 | 81.59 | 0 | 0 | 0 |
| Kodungaiyur, Chennai - TNPCB | wind_direction | 14340999 | deg | 12111 | 2025-02-19T01:00 | 2026-10-07T19:00 | 78.12 | 84.7 | 0 | 0 | 0 |
| Kodungaiyur, Chennai - TNPCB | wind_speed | 14341000 | m/s | 11763 | 2025-02-19T01:00 | 2026-09-10T18:00 | 75.87 | 86.18 | 0 | 0 | 0 |
| Manali Village, Chennai - TNPCB | co | 12235791 | ppb | 12826 | 2025-02-19T01:00 | 2026-10-07T19:00 | 82.73 | 89.7 | 0 | 0 | 0 |
| Manali Village, Chennai - TNPCB | no | 12235792 | ppb | 12724 | 2025-02-19T01:00 | 2026-10-07T19:00 | 82.07 | 88.99 | 0 | 0 | 0 |
| Manali Village, Chennai - TNPCB | no2 | 12235793 | ppb | 12721 | 2025-02-19T01:00 | 2026-10-07T16:00 | 82.05 | 88.98 | 0 | 0 | 0 |
| Manali Village, Chennai - TNPCB | nox | 14340913 | ppb | 12826 | 2025-02-19T01:00 | 2026-10-07T19:00 | 82.73 | 89.7 | 0 | 0 | 0 |
| Manali Village, Chennai - TNPCB | o3 | 12235794 | µg/m³ | 12106 | 2025-02-19T01:00 | 2026-09-18T10:00 | 78.08 | 87.51 | 0 | 0 | 0 |
| Manali Village, Chennai - TNPCB | pm10 | 12235795 | µg/m³ | 12659 | 2025-02-19T01:00 | 2026-10-07T18:00 | 81.65 | 88.54 | 0 | 0 | 0 |
| Manali Village, Chennai - TNPCB | pm25 | 12235796 | µg/m³ | 12730 | 2025-02-19T01:00 | 2026-10-07T18:00 | 82.11 | 89.03 | 0 | 0 | 0 |
| Manali Village, Chennai - TNPCB | relativehumidity | 12235797 | % | 12803 | 2025-02-19T01:00 | 2026-10-07T19:00 | 82.58 | 89.54 | 0 | 0 | 0 |
| Manali Village, Chennai - TNPCB | so2 | 12235798 | ppb | 12784 | 2025-02-19T01:00 | 2026-10-07T19:00 | 82.46 | 89.4 | 0 | 0 | 0 |
| Manali Village, Chennai - TNPCB | temperature | 12235799 | c | 12803 | 2025-02-19T01:00 | 2026-10-07T19:00 | 82.58 | 89.54 | 0 | 0 | 0 |
| Manali Village, Chennai - TNPCB | wind_direction | 14340914 | deg | 12803 | 2025-02-19T01:00 | 2026-10-07T19:00 | 82.58 | 89.54 | 0 | 0 | 0 |
| Manali Village, Chennai - TNPCB | wind_speed | 14340915 | m/s | 12803 | 2025-02-19T01:00 | 2026-10-07T19:00 | 82.58 | 89.54 | 0 | 0 | 0 |
| Manali, Chennai - CPCB | co | 12235648 | ppb | 12178 | 2025-02-19T01:00 | 2026-10-07T19:00 | 78.55 | 85.17 | 0 | 0 | 0 |
| Manali, Chennai - CPCB | no | 12235649 | ppb | 11869 | 2025-02-19T01:00 | 2026-10-07T19:00 | 76.55 | 83.01 | 0 | 0 | 0 |
| Manali, Chennai - CPCB | no2 | 12235650 | ppb | 11869 | 2025-02-19T01:00 | 2026-10-07T19:00 | 76.55 | 83.01 | 0 | 0 | 0 |
| Manali, Chennai - CPCB | nox | 14340870 | ppb | 11870 | 2025-02-19T01:00 | 2026-10-07T19:00 | 76.56 | 83.01 | 0 | 0 | 0 |
| Manali, Chennai - CPCB | o3 | 12235651 | µg/m³ | 12173 | 2025-02-19T01:00 | 2026-10-07T19:00 | 78.52 | 85.13 | 0 | 0 | 0 |
| Manali, Chennai - CPCB | pm10 | 12235652 | µg/m³ | 11811 | 2025-02-19T01:00 | 2026-10-07T19:00 | 76.18 | 82.6 | 0 | 0 | 0 |
| Manali, Chennai - CPCB | pm25 | 12235653 | µg/m³ | 11823 | 2025-02-19T01:00 | 2026-10-07T19:00 | 76.26 | 82.68 | 0 | 0 | 0 |
| Manali, Chennai - CPCB | relativehumidity | 12235654 | % | 12119 | 2025-02-19T01:00 | 2026-10-07T19:00 | 78.17 | 84.75 | 0 | 0 | 0 |
| Manali, Chennai - CPCB | so2 | 12235655 | ppb | 11804 | 2025-02-19T01:00 | 2026-10-07T19:00 | 76.14 | 82.55 | 0 | 0 | 0 |
| Manali, Chennai - CPCB | temperature | 12235656 | c | 10873 | 2025-02-19T01:00 | 2026-08-13T20:00 | 70.13 | 83.77 | 0 | 86 | 0 |
| Manali, Chennai - CPCB | wind_direction | 14340871 | deg | 11801 | 2025-02-19T01:00 | 2026-10-07T19:00 | 76.12 | 82.53 | 0 | 0 | 0 |
| Manali, Chennai - CPCB | wind_speed | 14340872 | m/s | 11801 | 2025-02-19T01:00 | 2026-10-07T19:00 | 76.12 | 82.53 | 0 | 0 | 0 |
| Perungudi, Chennai - TNPCB | co | 12236285 | ppb | 12574 | 2025-02-19T01:00 | 2026-09-29T00:00 | 81.1 | 89.25 | 0 | 0 | 0 |
| Perungudi, Chennai - TNPCB | no | 12236286 | ppb | 12541 | 2025-02-19T01:00 | 2026-09-29T00:00 | 80.89 | 89.02 | 0 | 0 | 0 |
| Perungudi, Chennai - TNPCB | no2 | 12236287 | ppb | 12556 | 2025-02-19T01:00 | 2026-09-29T00:00 | 80.99 | 89.13 | 0 | 0 | 0 |
| Perungudi, Chennai - TNPCB | nox | 14340992 | ppb | 12553 | 2025-02-19T01:00 | 2026-09-29T00:00 | 80.97 | 89.1 | 0 | 0 | 0 |
| Perungudi, Chennai - TNPCB | o3 | 12236288 | µg/m³ | 12535 | 2025-02-19T01:00 | 2026-09-29T00:00 | 80.85 | 88.98 | 0 | 0 | 0 |
| Perungudi, Chennai - TNPCB | pm10 | 12236289 | µg/m³ | 12533 | 2025-02-19T01:00 | 2026-09-29T00:00 | 80.84 | 88.96 | 0 | 0 | 0 |
| Perungudi, Chennai - TNPCB | pm25 | 12236290 | µg/m³ | 12499 | 2025-02-19T01:00 | 2026-09-29T00:00 | 80.62 | 88.72 | 0 | 0 | 0 |
| Perungudi, Chennai - TNPCB | relativehumidity | 12236291 | % | 11680 | 2025-02-19T01:00 | 2026-09-29T00:00 | 75.34 | 82.91 | 0 | 0 | 0 |
| Perungudi, Chennai - TNPCB | so2 | 12236292 | ppb | 12571 | 2025-02-19T01:00 | 2026-09-29T00:00 | 81.08 | 89.23 | 0 | 0 | 0 |
| Perungudi, Chennai - TNPCB | temperature | 12236293 | c | 11679 | 2025-02-19T01:00 | 2026-09-09T16:00 | 75.33 | 85.72 | 0 | 0 | 0 |
| Perungudi, Chennai - TNPCB | wind_direction | 14340993 | deg | 11678 | 2025-02-19T01:00 | 2026-09-29T00:00 | 75.32 | 82.89 | 0 | 0 | 0 |
| Perungudi, Chennai - TNPCB | wind_speed | 14340994 | m/s | 12196 | 2025-02-19T01:00 | 2026-09-29T00:00 | 78.66 | 86.57 | 0 | 0 | 0 |
| Royapuram, Chennai - TNPCB | co | 12236294 | ppb | 11487 | 2025-02-19T01:00 | 2026-10-07T19:00 | 74.09 | 80.33 | 0 | 0 | 0 |
| Royapuram, Chennai - TNPCB | no | 12236295 | ppb | 11309 | 2025-02-19T01:00 | 2026-10-07T19:00 | 72.94 | 79.09 | 0 | 0 | 0 |
| Royapuram, Chennai - TNPCB | no2 | 12236296 | ppb | 11363 | 2025-02-19T01:00 | 2026-10-07T19:00 | 73.29 | 79.47 | 0 | 0 | 0 |
| Royapuram, Chennai - TNPCB | nox | 14340995 | ppb | 11364 | 2025-02-19T01:00 | 2026-10-07T19:00 | 73.3 | 79.47 | 0 | 0 | 0 |
| Royapuram, Chennai - TNPCB | o3 | 12236297 | µg/m³ | 11089 | 2025-02-19T01:00 | 2026-10-07T19:00 | 71.52 | 77.55 | 0 | 0 | 0 |
| Royapuram, Chennai - TNPCB | pm10 | 12236298 | µg/m³ | 11464 | 2025-02-19T01:00 | 2026-10-07T19:00 | 73.94 | 80.17 | 0 | 0 | 0 |
| Royapuram, Chennai - TNPCB | pm25 | 12236299 | µg/m³ | 11387 | 2025-02-19T01:00 | 2026-10-07T19:00 | 73.45 | 79.63 | 0 | 0 | 0 |
| Royapuram, Chennai - TNPCB | relativehumidity | 12236300 | % | 11524 | 2025-02-19T01:00 | 2026-10-07T19:00 | 74.33 | 80.59 | 0 | 0 | 0 |
| Royapuram, Chennai - TNPCB | so2 | 12236301 | ppb | 11198 | 2025-02-19T01:00 | 2026-10-07T19:00 | 72.23 | 78.31 | 0 | 0 | 0 |
| Royapuram, Chennai - TNPCB | temperature | 12236302 | c | 10434 | 2025-02-19T01:00 | 2026-08-06T16:00 | 67.3 | 81.46 | 0 | 0 | 0 |
| Royapuram, Chennai - TNPCB | wind_direction | 14340996 | deg | 11524 | 2025-02-19T01:00 | 2026-10-07T19:00 | 74.33 | 80.59 | 0 | 0 | 0 |
| Royapuram, Chennai - TNPCB | wind_speed | 14340997 | m/s | 11501 | 2025-02-19T01:00 | 2026-10-07T19:00 | 74.18 | 80.43 | 0 | 0 | 0 |
| Velachery Res. Area, Chennai - CPCB | co | 12235526 | ppb | 12167 | 2025-02-19T01:00 | 2026-10-07T19:00 | 78.48 | 85.09 | 0 | 0 | 0 |
| Velachery Res. Area, Chennai - CPCB | no | 12235527 | ppb | 12094 | 2025-02-19T01:00 | 2026-10-07T19:00 | 78.01 | 84.58 | 0 | 0 | 0 |
| Velachery Res. Area, Chennai - CPCB | no2 | 12235528 | ppb | 12091 | 2025-02-19T01:00 | 2026-10-07T19:00 | 77.99 | 84.56 | 0 | 0 | 0 |
| Velachery Res. Area, Chennai - CPCB | nox | 14340852 | ppb | 12097 | 2025-02-19T01:00 | 2026-10-07T19:00 | 78.03 | 84.6 | 0 | 0 | 0 |
| Velachery Res. Area, Chennai - CPCB | o3 | 12235529 | µg/m³ | 12166 | 2025-02-19T01:00 | 2026-10-07T19:00 | 78.47 | 85.08 | 0 | 0 | 0 |
| Velachery Res. Area, Chennai - CPCB | pm10 | 12235530 | µg/m³ | 12149 | 2025-02-19T01:00 | 2026-10-07T19:00 | 78.36 | 84.96 | 0 | 0 | 0 |
| Velachery Res. Area, Chennai - CPCB | pm25 | 12235531 | µg/m³ | 12155 | 2025-02-19T01:00 | 2026-10-07T19:00 | 78.4 | 85.01 | 0 | 0 | 0 |
| Velachery Res. Area, Chennai - CPCB | relativehumidity | 12235532 | % | 11895 | 2025-02-19T01:00 | 2026-10-07T19:00 | 76.72 | 83.19 | 0 | 0 | 0 |
| Velachery Res. Area, Chennai - CPCB | so2 | 12235533 | ppb | 12151 | 2025-02-19T01:00 | 2026-10-07T19:00 | 78.37 | 84.98 | 0 | 0 | 0 |
| Velachery Res. Area, Chennai - CPCB | temperature | 12235534 | c | 10834 | 2025-02-19T01:00 | 2026-08-06T16:00 | 69.88 | 84.59 | 0 | 0 | 0 |
| Velachery Res. Area, Chennai - CPCB | wind_direction | 14340853 | deg | 10498 | 2025-02-19T01:00 | 2026-09-25T15:00 | 67.71 | 74.95 | 0 | 0 | 0 |
| Velachery Res. Area, Chennai - CPCB | wind_speed | 14340854 | m/s | 10502 | 2025-02-19T01:00 | 2026-09-25T15:00 | 67.74 | 74.98 | 0 | 0 | 0 |

Sensors whose metadata overlapped the window but returned **no valid hours**: 0 (none).

## Location × month coverage (hours with ≥1 valid pollutant value, %)

| location | 2025-01 | 2025-02 | 2025-03 | 2025-04 | 2025-05 | 2025-06 | 2025-07 | 2025-08 | 2025-09 | 2025-10 | 2025-11 | 2025-12 | 2026-01 | 2026-02 | 2026-03 | 2026-04 | 2026-05 | 2026-06 | 2026-07 | 2026-08 | 2026-09 | 2026-10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Alandur Bus Depot, Chennai - CPCB | 0 | 32 | 87 | 86 | 83 | 89 | 96 | 38 | 88 | 88 | 90 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Arumbakkam, Chennai - TNPCB | 0 | 33 | 38 | 0 | 0 | 60 | 99 | 98 | 95 | 96 | 92 | 89 | 67 | 82 | 93 | 95 | 90 | 77 | 99 | 90 | 56 | 76 |
| Chennai | 98 | 99 | 100 | 100 | 75 | 99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 95 | 99 | 100 | 100 | 100 | 100 |
| Kodungaiyur, Chennai - TNPCB | 0 | 3 | 87 | 80 | 81 | 90 | 86 | 93 | 88 | 95 | 93 | 84 | 67 | 83 | 95 | 94 | 89 | 88 | 99 | 90 | 53 | 76 |
| Manali Village, Chennai - TNPCB | 0 | 36 | 99 | 85 | 93 | 93 | 96 | 88 | 98 | 95 | 92 | 92 | 64 | 80 | 84 | 93 | 78 | 86 | 98 | 91 | 96 | 76 |
| Manali, Chennai - CPCB | 0 | 8 | 91 | 68 | 78 | 84 | 95 | 97 | 87 | 81 | 95 | 86 | 61 | 85 | 96 | 94 | 87 | 87 | 88 | 79 | 98 | 76 |
| Perungudi, Chennai - TNPCB | 0 | 35 | 99 | 87 | 94 | 92 | 95 | 95 | 98 | 81 | 88 | 96 | 66 | 86 | 95 | 95 | 94 | 88 | 98 | 90 | 51 | 0 |
| Royapuram, Chennai - TNPCB | 0 | 33 | 100 | 85 | 92 | 92 | 98 | 84 | 94 | 52 | 26 | 67 | 60 | 82 | 95 | 94 | 93 | 87 | 75 | 91 | 56 | 76 |
| Velachery Res. Area, Chennai - CPCB | 0 | 32 | 98 | 55 | 88 | 63 | 78 | 86 | 93 | 93 | 84 | 96 | 67 | 78 | 89 | 93 | 92 | 86 | 93 | 91 | 91 | 76 |

## Failed requests

None — every requested sensor-month was downloaded.

## Caveats

- Sensor metadata `coverage` fields are inconsistent (e.g. percentCoverage > 100%) and were not used.
- Several CPCB stations expose two sensors per gas: a legacy series in µg/m³ and a newer series in ppb. They are kept separate here; no unit conversion is applied in this audit.
- Location coordinates are as reported by OpenAQ. Some are inconsistent with the station name (see final_data_audit.md for the station-to-zone reconciliation).
- OpenAQ hourly bins start at :30 UTC (IST hour boundaries); hours are reported in IST.
