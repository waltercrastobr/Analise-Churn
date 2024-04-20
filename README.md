# Análise de Churn de Funcionários

<img src="https://github.com/waltercrastobr/Analise-Churn/blob/main/img_churn.jpg" alt="Descrição da imagem">


Este repositório contém uma análise de dados completa para prever a probabilidade de um funcionário deixar uma empresa, com foco em estratégias de retenção de funcionários. O projeto utiliza uma [base](https://github.com/waltercrastobr/Analise-Churn/blob/main/Human_Resources.csv) de dados fictícia da IBM (International Business Machine Corporation), disponível no [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
, e segue uma abordagem de aprendizado supervisionado para classificação. 

## Introdução

### Contexto do Projeto

Neste projeto, foi construído um modelo de machine learning para prever a probabilidade de um funcionário sair de uma empresa. Como dito anteriormente, utilizou-se uma base de dados fictícia fornecida pela IBM, disponível no site Kaggle. A análise envolveu aprendizado supervisionado, utilizando um conjunto de treinamento rotulado para classificação, onde o alvo é 1 se o funcionário sair, caso contrário, 0.

### Problema de Negócio

Um gerente da empresa está preocupado com o crescente número de funcionários deixando a empresa. Eles desejam prever quão provável é que um funcionário saia, para que possam abordá-los proativamente e oferecer melhores condições de trabalho, visando reverter as decisões dos funcionários na direção oposta.

### Objetivos do Projeto

1. Identificar os fatores associados à saída de funcionários.
2. Construir um modelo preciso para prever a probabilidade de um funcionário sair.
3. Oferecer planos de ação para reduzir a saída de funcionários.

### Benefícios Esperados

- Modelo de machine learning capaz de prever com precisão a probabilidade de churn.
- Sugestão de ações para a empresa reduzir a saída de funcionários.
- Alta taxa de previsão correta, especialmente em relação aos funcionários propensos ao churn.
- Melhoria na retenção de funcionários e na experiência do funcionário.
- Proteção de bons funcionários e da receita da empresa.

### Meta de Desempenho

A meta é alcançar uma taxa de 70% de identificação dos casos de Churn, visando reduzir a taxa de saída de funcionários da empresa.

Este relatório apresenta a metodologia, resultados e discussões relevantes para o projeto de análise de dados e construção do modelo de machine learning.

## Bibliotecas Usadas
Abaixo estão as principais bibliotecas utilizadas neste projeto:

- pandas: Para manipulação de dados tabulares.
- numpy: Para operações matemáticas e manipulação de arrays.
- seaborn: Para visualização de dados estatísticos.
- matplotlib: Para criação de gráficos e visualizações.
- scikit-learn: Para construção e avaliação de modelos de machine learning.
- imbalanced-learn: Para lidar com desbalanceamento de classes em conjuntos de dados.
- xgboost: Para implementação do algoritmo XGBoost.
- SMOTE: Para a técnica de oversampling SMOTE.
- NearMiss: Para a técnica de undersampling NearMiss.
  
Além disso, foram utilizados os seguintes modelos de machine learning:

- LogisticRegression
- RandomForestClassifier
- AdaBoostClassifier
- XGBClassifier
- SVC
- KNeighborsClassifier
- GaussianNB

## Dicionários de Atributos
#### **Target (Alvo)**

- **Attrition**: Indica se o funcionário deixou a empresa (Yes: 1) ou permaneceu (No: 0).

#### **Dados Categóricos**:

- **BusinessTravel**: Nível de viagem de negócios do funcionário (Baixo, Médio, Alto).
- **Department**: Departamento em que o funcionário trabalha (Vendas, Pesquisa e Desenvolvimento, RH).
- **EducationField**: Área de educação do funcionário (Ciências da Vida, Marketing, Medicina, etc.).
- **Gender**: Gênero do funcionário (Masculino, Feminino).
- **JobRole**: Cargo do funcionário (Gerente, Representante de Vendas, Técnico de Laboratório, etc.).
- **MaritalStatus**: Estado civil do funcionário (Solteiro, Casado, Divorciado).

#### **Dados Numéricos**:

- **Age**: Idade do funcionário.
- **DailyRate**: Taxa diária de pagamento do funcionário.
- **DistanceFromHome**: Distância da casa do funcionário até o local de trabalho.
- **Education**: Nível de educação do funcionário (1: Abaixo da faculdade, 2: - Faculdade, 3: Bacharel, 4: Mestre, 5: Doutor).
- **EnvironmentSatisfaction**: Nível de satisfação com o ambiente de trabalho (1: Baixo, 2: Médio, 3: Alto, 4: Muito Alto).
- **HourlyRate**: Taxa de pagamento por hora do funcionário.
- **JobInvolvement**: Nível de envolvimento com o trabalho (1: Baixo, 2: Médio, 3: Alto, 4: Muito Alto).
- **JobLevel**: Nível do cargo do funcionário.
- **JobSatisfaction**: Nível de satisfação com o trabalho (1: Baixo, 2: Médio, 3: Alto, 4: Muito Alto).
- **MonthlyIncome**: Renda mensal do funcionário.
- **MonthlyRate**: Taxa mensal de pagamento do funcionário.
- **NumCompaniesWorked**: Número de empresas em que o funcionário trabalhou anteriormente.
- **OverTime**: Se o funcionário trabalha horas extras (Sim, Não).
- **PercentSalaryHike**: Aumento percentual no salário do funcionário.
- **PerformanceRating**: Avaliação de desempenho do funcionário (1: Baixo, 2: Bom, 3: Excelente, 4: Excepcional).
- **RelationshipSatisfaction**: Nível de satisfação com o relacionamento no trabalho (1: Baixo, 2: Médio, 3: Alto, 4: Muito Alto).
- **StockOptionLevel**: Nível de opção de compra de ações do funcionário.
- **TotalWorkingYears**: Total de anos trabalhados pelo funcionário.
- **TrainingTimesLastYear**: Número de vezes que o funcionário foi treinado no ano passado.
- **WorkLifeBalance**: Equilíbrio entre trabalho e vida pessoal do funcionário (1: Ruim, 2: Bom, 3: Melhor, 4: Ótimo).
- **YearsAtCompany**: Número de anos que o funcionário trabalhou na empresa atual.
- **YearsInCurrentRole**: Número de anos que o funcionário está no cargo atual.
- **YearsSinceLastPromotion**: Número de anos desde a última promoção do funcionário.
- **YearsWithCurrManager**: Número de anos trabalhando com o gerente atual.
- **EmployeeCount** (Contagem de Funcionários): Número de funcionários.
- **StandardHours** (Horas Padrão): Número padrão de horas trabalhadas.
- **Over18** (Acima de 18 anos): Indicação se o funcionário é maior de 18 anos.
- **EmployeeNumber** (Número do Funcionário): Número identificador único para cada funcionário.

# Notebooks

## [Análise Exploratória de Dados](https://github.com/waltercrastobr/Analise-Churn/blob/main/EDA_Analise_Churn.ipynb)
- **Objetivo:** Desenvolver uma análise exploratória de dados para obtenção de insights valiosos e preparação da base de dados para criação de modelos preditivos.
  1. Analise e Tratamento dos dados.
  2. Insights gerados a partir dos dados.
  3. Pré Processamento dos dados e dividao entre treino e teste..
     
- **Resultados:**
Realizei a limpeza, tratamento e transformação dos dados, convertendo variáveis categóricas em numéricas e padronizando os valores. A base de dados estava livre de outliers, dados nulos e duplicados, o que simplificou o processo. Em seguida, criei gráficos comparativos entre os funcionários que saíram e os que permaneceram na empresa para obter insights, documentando essas descobertas em um relatório no notebook. Posteriormente, dividi os dados em conjuntos de treino e teste e os preparei para a criação dos modelos preditivos.

## [Modelos Preditivos](https://github.com/waltercrastobr/Analise-Churn/blob/main/Modelos_Preditivos__An%C3%A1lise__Churn.ipynb)

- **Objetivo:** Desenvolver um modelo preditivo preciso para prever a probabilidade de churn de funcionários.
- **Passos:**
  1. Construção do modelo de machine learning.
  2. Treinamento e avaliação do modelo.
  3. Seleção dos melhores modelos para tentativa de otimizá-los.
 
- **Resultados:**
Foram inicialmente testados os modelos regressão logística e no random forest devido à sua capacidade com dados desbalanceados. Após avaliar seus desempenhos, outros modelos foram considerados para melhorar as métricas de recall e F1-score, cruciais na análise de churn. Além da regressão logística e do XGB, foram testados o AdaBoostClassifier, SVC, KNeighborsClassifier e GaussianNB. Os melhores resultados foram obtidos com a regressão logística e o XGB, levando a uma otimização desses modelos para maximizar seu desempenho.

## [Otimização de Modelos](https://github.com/waltercrastobr/Analise-Churn/blob/main/Otimizacao_Modelos__An%C3%A1lise__Churn.ipynb)

- **Objetivo:** Aprimorar os modelos existentes, selecionando os melhores atributos e lidando com o desbalanceamento de classes.
- **Passos:**
  1. Tentativa de Seleção de atributos relevantes.
  2. Tratamento do desbalanceamento de classes com Oversampling e Undersampling.
  3. Seleção do modelo final.
 
- **Resultados:**
Inicialmente, foi realizada uma seleção de atributos na tentativa de melhorar o desempenho do modelo, porém não houve melhora significativa, possivelmente devido à perda de alguns dados importantes durante o processo. Em seguida, foi identificado que o desbalanceamento de classes poderia ser o principal problema. Para lidar com isso, foram utilizadas técnicas de oversampling e undersampling. Após a aplicação dessas técnicas e a verificação de possíveis problemas de overfitting, foi constatado que o modelo de regressão logística com oversampling (SMOTE) apresentou o melhor desempenho.

## [Salvando o Classificador](https://github.com/waltercrastobr/Analise-Churn/blob/main/Classificador_Salvo_An%C3%A1lise__Churn.ipynb)

- **Objetivo:** Salvar o melhor classificador obtido e testá-lo para verificar sua capacidade de prever corretamente o churn de funcionários.
- **Passos:**
  1. Salvamento do classificador.
  2. Carregamento e teste do modelo salvo.
 
- **Resultados:**
O modelo foi salvaguardado com sucesso e submetido a testes bem-sucedidos, utilizando o método pickle para sua preservação. Demonstrou uma capacidade de previsão sólida em relação à probabilidade de saída de funcionário, alcançando 80% de recall para a classe alvo e 77% de F1-score para a mesma classe. O modelo está disponível para acesso e uso através deste [link](https://github.com/waltercrastobr/Analise-Churn/blob/main/modelo_rl_smote.pkl), representando uma ferramenta valiosa para a empresa na identificação e retenção de seus colaboradores.

## [Relatório de Resultados Financeiros](https://github.com/waltercrastobr/Analise-Churn/blob/main/relatorio_ganhos.pdf)
Em conclusão, o uso do modelo resultou em uma economia significativa de custos e tempo para a empresa, em comparação com o cenário sem o modelo. Em cinco meses, a diferença total foi de $2,983,440, considerando os custos de contratação (reposição dos 'churners') e o lucro acumulado. 















