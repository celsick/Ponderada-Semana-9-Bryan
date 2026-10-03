# Autoestudo 3 — Tópicos Avançados em PLN

- **Nome:** Celso Rodrigues Rocha Júnior
- **Curso:** Engenharia de Software
- **Módulo:** 07
- **Professor:** Bryan Kano Ferreira
- **Aula realizada:** Tópicos Avançados em Processamento de Linguagem Natural

## Sobre a atividade

Este repositório contém a entrega do Autoestudo 3 na alternativa de **aprendizado contínuo**. O trabalho propõe uma arquitetura para que um sistema conversacional seja atualizado de forma controlada ao longo do tempo, diante de mudanças no ambiente de uso e do problema de *concept drift*.

Os conceitos centrais são:

- **Atualização de conhecimento:** distinguir fatos que mudaram, conhecimento novo e conhecimento que deve ser preservado, e decidir quando basta atualizar a base de conhecimento e quando é preciso adaptar o modelo.
- **Concept drift:** detectar mudanças na distribuição das entradas ou na relação entre entradas e saídas, tratando cada alerta como hipótese a investigar, e não como ordem automática de retreinamento.
- **Avaliação de versões:** comparar cada versão candidata com a versão em produção em dados recentes, históricos e de segurança antes de qualquer implantação, com implantação gradual e *rollback*.
- **Prevenção do esquecimento catastrófico:** usar conjuntos de retenção e testes de regressão para garantir que o aprendizado do novo não apague capacidades anteriores.

A proposta também incorpora os temas de segurança discutidos na disciplina — ataques a modelos de linguagem, envenenamento de dados, proteção de dados pessoais e gestão de riscos de IA segundo publicações do NIST — como parte do próprio processo de aprendizado contínuo.

## Conteúdo do repositório

| Arquivo | Finalidade |
|---|---|
| [README.md](README.md) | Apresentação da atividade, dados do estudante e guia do repositório |
| [proposta-aprendizado-continuo.md](proposta-aprendizado-continuo.md) | Documento acadêmico completo: introdução, solução proposta com diagrama de arquitetura, conclusão e referências em formato ABNT |

O documento principal está organizado em:

1. [Introdução](proposta-aprendizado-continuo.md#1-introdução): problema, diferença entre *concept drift* e desatualização do conhecimento, esquecimento catastrófico e riscos de segurança da atualização.
2. [Solução proposta](proposta-aprendizado-continuo.md#2-solução-proposta): princípios de projeto, diagrama de arquitetura, responsabilidades dos módulos, ciclo de atualização, avaliação, segurança e governança, e viabilidade de implementação.
3. [Conclusão](proposta-aprendizado-continuo.md#3-conclusão): síntese crítica, limitações e considerações sobre o esforço de implementação.
4. [Referências](proposta-aprendizado-continuo.md#referências): fontes citadas, em formato ABNT.

## Visualização

O diagrama de arquitetura foi escrito em [Mermaid](https://mermaid.js.org/) diretamente no arquivo Markdown, sem imagens externas. No GitHub, ele é renderizado automaticamente ao abrir o [documento principal](proposta-aprendizado-continuo.md#22-diagrama-de-arquitetura). Em editores locais, a visualização depende de suporte a Mermaid no visualizador de Markdown utilizado.
