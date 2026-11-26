# SCRIPT 02: Causal Elasticity & Counterfactual Demand Curve

library(tidyverse)
set.seed(1897)

# ==============================================================================
# 1. Cargar datos observables

ecommerce_panel <- read.csv("data/ecommerce_panel.csv")

# Filtrando SKU101 (con elasticidad poblacional cerca a a -2.15
sku1_df <- ecommerce_panel %>% 
  filter(sku_id == "SKU_101")

# ==============================================================================
# 2. Modelo de Poisson ingenuo

modelo_ingenuo <- glm(
  units_sold ~ log(effective_price) + log(competitor_price) + 
    is_weekend + is_peak_event,
  family = poisson(link = "log"),
  data = sku1_df)

# ==============================================================================
# 3. Modelo de Poisson con Control Function (IV: supplier_freight_index)

# Primero aislamos la variación exógena del log-precio vía fletes del proveedor
etapa_1 <- lm(
  log(effective_price) ~ supplier_freight_index + log(competitor_price) + 
    is_weekend + is_peak_event,
  data = sku1_df
)

sku1_df <- sku1_df %>% 
  mutate(iv_residuals = residuals(etapa_1))

# Segundo incluimos iv_residuals controla por el demand_shock no observable
modelo_causal <- glm(
  units_sold ~ log(effective_price) + log(competitor_price) + is_weekend +
    is_peak_event + iv_residuals,
  family = poisson(link = "log"),
  data = sku1_df
)

# Resumen comparativo de elasticidades
elast_ingenua <- coef(modelo_ingenuo)["log(effective_price)"]
elast_causal <- coef(modelo_causal)["log(effective_price)"]

cat("\n--- Comparación de Elasticidades (SKU_101 | Real = -2.15) ---\n")
cat("Poisson Ingenuo:          ", round(elast_ingenua, 4), "\n")
cat("Control Function Poisson: ", round(elast_causal, 4), "\n")

# ==============================================================================
# 4. Bootstrap para error estándar e intervalo de confianza de elasticidad causal 

B <- 300
boot_elast <- numeric(B)

for (b in seq_len(B)) {
  idx <- sample(seq_len(nrow(sku1_df)), size = nrow(sku1_df), replace = T)
  d_b <- sku1_df[idx, ]
  
  m1_b <- lm(
    log(effective_price) ~ supplier_freight_index + log(competitor_price) + 
      is_weekend + is_peak_event,
    data = d_b
  )
  d_b$res_b <- residuals(m1_b)
  
  m2_b <- glm(
    units_sold ~ log(effective_price) + log(competitor_price) + is_weekend + 
      is_peak_event + res_b,
    family = poisson(link = "log"),
    data = d_b
  )
  boot_elast[b] <- coef(m2_b)["log(effective_price)"]
  
}

ic_95 <- quantile(boot_elast, probs = c(0.025, 0.975))
cat("IC 95% Bootstrap Elasticidad Causal: [", round(ic_95[1], 4), ", ", round(ic_95[2], 4), "]\n")

summary(boot_elast)
head(boot_elast, 10)
sum(boot_elast == 0)

# ==============================================================================
# 5. Curva de Demanda Contrafactual Q_hat(P) en un día 

# Fijamos variables de control en escenario típido: día hábil, precio de competencia
# en su promedio y residuo IV = 0 (sin shock)
precio_comp_ref <- mean(sku1_df$competitor_price, na.rm = T)

grilla_precios <- tibble(
  effective_price = seq(60, 200, by = 2),
  competitor_price = precio_comp_ref,
  is_weekend = 0L,
  is_peak_event = 0L,
  iv_residuals = 0
) %>% mutate(
  q_hat_causal = predict(modelo_causal, newdata = ., type = "response"),
  q_hat_ingenuo = predict(modelo_ingenuo, newdata = ., type = "response")
)

# Guardamos la grilla de demanda para después contrastarla con Python ML
dir.create("outputs", showWarnings = F)
write_csv(grilla_precios, "outputs/demanda_contrafactual_causal_sku101.csv")

# Gráfico
grafico_curva <- ggplot(grilla_precios, aes(
  x = effective_price, y = q_hat_causal)) + 
  geom_line(linewidth = 1.1) + 
  geom_vline(xintercept = mean(sku1_df$effective_price), linetype = "dashed", alpha = 0.7) +
  labs(
    title = "Curva de Demanda Contrafactual Causal - SKU_101 (Auriculares ANC Pro)",
    subtitle = paste0(
      "Elasticidad estimada (Control function): ", round(elast_causal, 3),
      " | Línea punteada: precio promedio histórico"
    ),
    x = "Precio Efectivo ($ USD)", y = "Unidades Esperadas Diarias Q_hat(P)"
  ) + theme_minimal(base_size = 12)

print(grafico_curva)  
ggsave("outputs/curva_demanda_causal_sku101.png", plot = grafico_curva, width = 8, height = 5, dpi = 300)  
  
 
  
  
