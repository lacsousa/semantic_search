"""Mapeamento PROPOSTO de group (Enare RM) -> macroárea. Aprovado pelo usuário (Etapa 1)."""
MACROAREAS = {
    "Clínica Médica": [
        "Clínica Médica", "Cardiologia", "Hematologia e Hemoterapia", "Neurologia", "Nefrologia",
        "Infectologia", "Gastroenterologia", "Endocrinologia e Metabologia", "Oncologia", "Pneumologia",
        "Alergia e Imunologia", "Reumatologia", "Dermatologia", "Medicina Intensiva", "Medicina de Emergência",
        "Psiquiatria",
    ],
    "Cirurgia": [
        "Cirurgia Geral", "Cirurgia Plástica", "Cirurgia Torácica", "Cirurgia Vascular e Angiologia",
        "Cirurgia do Aparelho Digestivo", "Cirurgia de Cabeça e Pescoço", "Neurocirurgia",
        "Ortopedia e Traumatologia", "Anestesiologia",
    ],
    "Pediatria": ["Pediatria", "Medicina do Adolescente"],
    "Ginecologia e Obstetrícia": ["Ginecologia e Obstetrícia"],
    "Saúde Coletiva / Medicina de Família": ["Medicina de Família, Preventiva e Saúde Coletiva"],
    "Outras": [
        "Radiologia e Diagnóstico por Imagem", "Homeopatia", "Patologia", "Oftalmologia", "Foniatria",
        "Otorrinolaringologia", "Endoscopia Digestiva", "Medicina Física e Reabilitação",
    ],
}
GROUP_PARA_MACRO = {g: m for m, gs in MACROAREAS.items() for g in gs}

# Nomes manuais dos clusters K-Means (k=8, versão a) — aprovados pelo usuário (Etapa 3)
NOMES_CLUSTERS = {
    0: "Pediatria: febre e infecções na criança",
    1: "Ginecologia: ciclo, contracepção, corrimento",
    2: "Clínica: exames laboratoriais",
    3: "Adulto: urgência, cirurgia e urologia (misto)",
    4: "Nódulos: mama e tireoide / idosos (misto)",
    5: "Obstetrícia: pré-natal e parto",
    6: "Saúde coletiva e atenção primária",
    7: "Pediatria: lactentes e puericultura",
}
