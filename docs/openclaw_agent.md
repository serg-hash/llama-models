# Crear e introducir un agente en OpenClaw

Esta guía describe un flujo mínimo para crear un agente y registrarlo en OpenClaw.

## 1) Definir el perfil del agente

Crea un archivo de configuración para el agente (por ejemplo `openclaw/agents/soporte_llama.yaml`):

```yaml
id: soporte-llama
name: Soporte Llama
model: meta-llama/Meta-Llama-3.1-8B-Instruct
system_prompt: |
  Eres un agente de soporte técnico especializado en despliegues con Llama.
  Responde en español, de forma precisa y accionable.
tools:
  - search_docs
  - ticket_creator
policies:
  - safe_output
  - pii_redaction
```

Campos recomendados:

- `id`: identificador único del agente.
- `name`: nombre legible para UI y logs.
- `model`: modelo Llama a utilizar.
- `system_prompt`: comportamiento base del agente.
- `tools`: herramientas que puede invocar.
- `policies`: controles de seguridad obligatorios.

## 2) Registrar el agente en OpenClaw

En el archivo de registro (por ejemplo `openclaw/agents/registry.yaml`) agrega la entrada:

```yaml
agents:
  - id: soporte-llama
    config: agents/soporte_llama.yaml
    enabled: true
```

## 3) Exponerlo en el router de tareas

Enruta tipos de solicitud al nuevo agente:

```yaml
routes:
  - match:
      domain: soporte
      language: es
    agent_id: soporte-llama
```

## 4) Validar el comportamiento

Antes de publicarlo en producción:

1. Prueba prompts felices y ambiguos.
2. Verifica denegación segura ante peticiones riesgosas.
3. Comprueba observabilidad (`trace_id`, latencia, herramientas invocadas).
4. Activa rollout progresivo (por ejemplo 5% -> 25% -> 100%).

## 5) Prompt de humo para QA

Usa este prompt de verificación rápida:

```text
Necesito ayuda para desplegar Meta-Llama-3.1-8B-Instruct en GPU A10.
Dame pasos, riesgos y una checklist final.
```

Resultado esperado:

- Respuesta en español.
- Pasos concretos.
- Riesgos operativos y de seguridad.
- Checklist final accionable.
