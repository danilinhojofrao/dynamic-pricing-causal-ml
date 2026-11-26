# ==============================================================================
# PROYECTO: Dynamic Price Optimization & Causal Elasticity Estimation (CORREGIDO)
# ==============================================================================

library(tidyverse)
library(lubridate)

set.seed(1897)

# 1. Parámetros Estructurales (Ajustamos true_alpha para tener un volumen realista > 50 uds/día)
skus <- tibble(
  sku_id = c("SKU_101", "SKU_202", "SKU_303"),
  product_name = c("Auriculares ANC Pro", "Teclado Mecánico RGB", "Hub USB-C 7-en-1"),
  category = c("Audio", "Periféricos", "Accesorios"),
  base_price = c(120.0, 85.0, 35.0),
  base_cogs = c(52.0, 34.0, 11.0),
  fulfillment_cost = c(4.5, 5.0, 3.0),
  true_alpha = c(11.5, 10.2, 9.5),           # Nivel base realista (~75 uds/día para SKU_101)
  true_price_elast = c(-2.15, -1.75, -1.25), # Elasticidad precio propia real
  true_cross_elast = c(0.65, 0.45, 0.25)     # Elasticidad cruzada vs competencia
)

# 2. Calendario de 2 años (730 días)
dates <- seq(as.Date("2024-01-01"), as.Date("2025-12-31"), by = "day")

calendar_df <- tibble(date = dates) %>%
  mutate(
    day_of_week = wday(date, label = TRUE),
    is_weekend = if_else(wday(date) %in% c(1, 7), 1, 0),
    month = month(date),
    is_peak_event = if_else(
      (month == 5 & day(date) %in% 25:28) |
        (month == 10 & day(date) %in% 2:5) |
        (month == 12 & day(date) %in% 15:24), 1, 0
    ),
    # Variable Instrumental: Choque exógeno de fletes de nuestro proveedor exclusivo
    freight_shock = arima.sim(model = list(ar = 0.85), n = n(), sd = 0.12) %>% as.numeric(),
    supplier_freight_index = 1 + freight_shock
  )

# 3. Construcción de la Tabla Analítica Base
ecommerce_df <- crossing(calendar_df, skus) %>%
  arrange(sku_id, date) %>%
  group_by(sku_id) %>%
  mutate(
    unit_cogs = round(base_cogs * supplier_freight_index, 2),
    marginal_cost = unit_cogs + fulfillment_cost,
    
    # Choque de demanda no observable (se genera ANTES porque el precio reacciona a él)
    demand_shock = rnorm(n(), mean = 0, sd = 0.15),
    
    # Precio de la competencia: tiene su propia dinámica (evita colinealidad con nuestro IV)
    comp_noise = arima.sim(model = list(ar = 0.7), n = n(), sd = 0.06) %>% as.numeric(),
    competitor_price = round(base_price * (1 + comp_noise), 2),
    
    # --- ENDOGENEIDAD REAL DEL PRECIO ---
    # El equipo comercial observa parte del demand_shock (0.35 * demand_shock) y sube precios,
    # generando sesgo positivo en OLS/Poisson ingenuo. Además traspasa el costo de flete (IV).
    commercial_markup = 0.04 * is_weekend + 0.08 * is_peak_event + 
      0.50 * freight_shock + 0.35 * demand_shock,
    
    promo_discount = case_when(
      is_peak_event == 1 ~ runif(n(), 0.15, 0.25),
      runif(n()) < 0.08 ~ runif(n(), 0.05, 0.12),
      TRUE ~ 0
    ),
    
    # Precio efectivo
    effective_price = round(base_price * (1 + commercial_markup) * (1 - promo_discount) * exp(rnorm(n(), 0, 0.02)), 2),
    discount_pct = round(pmax(0, (base_price - effective_price / base_price), 4)),
    
    # --- FUNCIÓN DE DEMANDA ESTRUCTURAL ---
    log_expected_q = true_alpha +
      true_price_elast * log(effective_price) +
      true_cross_elast * log(competitor_price) +
      0.22 * is_weekend +
      0.55 * is_peak_event +
      demand_shock,
    
    units_sold = rpois(n(), lambda = exp(log_expected_q)),
    
    # Resultados financierios diarios
    revenue = round(effective_price * units_sold, 2),
    total_cost = round(marginal_cost * units_sold, 2),
    gross_margin = round(revenue - total_cost, 2)
  ) %>%
  ungroup()

# 4. Verificación de Estimadores en SKU_101 (Ground Truth = -2.15)
sku1_completo <- ecommerce_df %>% filter(sku_id == "SKU_101")

# A) Modelo Ingenuo (Sesgado porque effective_price correlaciona con demand_shock)
poisson_ingenuo <- glm(
  units_sold ~ log(effective_price) + log(competitor_price) + is_weekend + is_peak_event,
  family = poisson(link = "log"),
  data = sku1_completo
)

# B) Función de Control (2 Etapas con Variable Instrumental supplier_freight_index)
etapa_1 <- lm(
  log(effective_price) ~ supplier_freight_index + log(competitor_price) + is_weekend + is_peak_event,
  data = sku1_completo
)
sku1_completo$iv_residuals <- residuals(etapa_1)

etapa_2_cf <- glm(
  units_sold ~ log(effective_price) + log(competitor_price) + is_weekend + is_peak_event + iv_residuals,
  family = poisson(link = "log"),
  data = sku1_completo
)

cat("\n--- Elasticidad Poisson Ingenuo (Sesgada hacia arriba por endogeneidad) ---\n")
print(round(coef(poisson_ingenuo)["log(effective_price)"], 3))

cat("\n--- Elasticidad Recuperada con Control Function IV (Target = -2.15) ---\n")
print(round(coef(etapa_2_cf)["log(effective_price)"], 3))

# Dataset observable para modelación y optimización 
ecommerce_panel <- ecommerce_df %>% 
  select(
    date, sku_id, product_name, category, is_weekend, is_peak_event,
    units_sold, base_price, effective_price, discount_pct, competitor_price,
    unit_cogs, fulfillment_cost, marginal_cost, supplier_freight_index,
    revenue, total_cost, gross_margin
  )

# 5. Exportación a CSV para compartir con Python

dir.create("data", showWarnings = F)
write_csv(ecommerce_panel, "data/ecommerce_panel.csv")
cat("Dataset exportado exitosamente en: data/ecommerce_panel.csv\n")
cat("Dimensiones:", nrow(ecommerce_panel), "filas x", ncol(ecommerce_panel), "columnas\n")
glimpse(ecommerce_panel)




  
