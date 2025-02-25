# Private Synthetic Data Generation Made Easy: Out-of-the-Box with Docker, Argilla & Ollama

> "Empowering organizations with a turnkey solution for synthetic dataset creation in private environments."

The increasing adoption of AI solutions across industries has created an unprecedented demand for high-quality training data. As organizations scale their AI initiatives, they face the dual challenge of generating substantial, domain-specific datasets while ensuring data privacy and security. Traditional approaches often involve compromises: either using public datasets that may not fully align with specific needs, or investing heavily in custom data generation infrastructure.

The complexity of this challenge is amplified by regulatory requirements, resource constraints, and the need for specialized expertise. Organizations must navigate GDPR, CCPA, and industry-specific regulations while maintaining efficient data generation pipelines. This has created a pressing need for solutions that can operate entirely within private infrastructure while maintaining enterprise-grade capabilities.

## The Challenge

The development of AI models requires extensive training data, yet organizations face significant obstacles in data generation and management. Privacy regulations and security requirements often prevent the use of public datasets or cloud-based generation services. Additionally, existing solutions typically demand complex infrastructure setups and significant technical expertise, increasing both implementation time and costs.

Modern enterprises require a solution that addresses several critical aspects:
1. Data Privacy: Complete control over data generation and storage
2. Infrastructure Flexibility: Deployment options that fit existing systems
3. Quality Assurance: Tools for data validation and curation
4. Scalability: Ability to grow with increasing data needs
5. Cost Efficiency: Reduction in infrastructure and maintenance costs

## The Solution

This out-of-the-box Synthetic Dataset Generator approach leverages the power of three technologies to create a seamless, private data generation pipeline. At its core is the [Synthetic Dataset Generator](https://github.com/argilla-io/synthetic-data-generator), a tool designed for dataset creation. [Ollama](https://ollama.ai/) ensures secure local LLM inference with [Distilabel](https://github.com/argilla-io/distilabel) integration, while [Argilla's](https://argilla.io/) data curation capabilities complete the workflow, all operating within your secure infrastructure.

This architecture delivers key technical advantages:
- Full data sovereignty with containerized local deployment
- End-to-end pipeline from generation to validation
- Modular design for system integration

Here's how it all fits together:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64461026e1fd8d65b27e6187/Uz-kDOBrV-_GahUrc1K_O.png)

The Docker-based architecture integrates these core services:

**[Synthetic Dataset Generator](https://github.com/argilla-io/synthetic-data-generator)**: No-code interface for dataset creation
**[Ollama](https://github.com/ollama/ollama)**: Local inference with Distilabel for structured generation
**[Argilla](https://github.com/argilla-io/argilla)**: Comprehensive data curation platform

Let's explore how these components work together in a practical workflow.

## 1. Installation & Setup



### 1.1 Clone Repository
```bash
git clone https://github.com/argilla-io/synthetic-data-generator
cd synthetic-data-generator
```

### 1.2 Environment Setup
```bash
# Copy environment template
cp docker/.env.docker.template .env

# Model configuration in .env (if using Ollama)
MODEL="deepseek-r1:1.5b"  # Must match Ollama model name
```

### 1.3 Build & Deploy Services
> Pro tip: Even if you're planning to use just one component initially, we recommend building all services to enable future functionality without rebuilding. For detailed deployment options, check the [Docker documentation](https://github.com/argilla-io/synthetic-data-generator/blob/main/docker/README.md).

> Note: Ollama runs on CPU/GPU for Linux/Windows in Docker. For macOS, only CPU is supported in Docker - for GPU support, install Ollama separately ([details](https://ollama.com/blog/ollama-is-now-available-as-an-official-docker-image)).

```bash
# Build all services
docker compose -f docker-compose.yml -f docker/ollama/compose.yml -f docker/argilla/compose.yml build
# Start all services
docker compose -f docker-compose.yml -f docker/ollama/compose.yml -f docker/argilla/compose.yml up -d
```

To view logs, either:
- Use Docker Desktop's interface
- Remove the `-d` flag when running the above command
- Execute the following for specific service logs:
  ```bash
  # Core App logs
  docker compose logs -f app
  # Ollama logs
  docker compose -f docker-compose.yml -f docker/ollama/compose.yml logs -f ollama
  # Argilla logs
  docker compose -f docker-compose.yml -f docker/argilla/compose.yml logs -f argilla
  ```

## 2. Dataset Generation

Let's walk through creating a RAG dataset. For a detailed overview of the generation process, check out the [introduction to the Synthetic Data Generator](https://huggingface.co/blog/synthetic-data-generator).

2.1. **Dataset Description**
   ```text
   A dataset to retrieve information from information security policies
   ```

   System initializes and processes the prompt:
   ![image/png](https://cdn-uploads.huggingface.co/production/uploads/64461026e1fd8d65b27e6187/sxH8JChF-HnGMOilymYpA.png)


2.2. **Task Configuration & Sample Generation**
   System analyzes and generates the system prompt and optimal parameters automatically. Then, samples are generated for validation (modify system prompt or parameters manually if needed, then click save to generate sample data):
   ![image/png](https://cdn-uploads.huggingface.co/production/uploads/64461026e1fd8d65b27e6187/mYVlGNnz6YNrPJutxmBtR.png)


2.3. **Full Dataset Generation**
   After validating the sample data quality, proceed with full dataset generation. Configure the following parameters:

   - **Repository Owner**: Your Hugging Face username for dataset hosting
   - **Dataset Name**: A descriptive name following standard naming conventions
   - **Number of Examples**: Define dataset size (recommended: 100-1000 for initial deployments)
   - **Temperature**: Controls generation creativity (default 0.7 balances coherence and diversity)
   - **Privacy Settings**: Optional dataset privacy configuration for Hugging Face Hub

   The temperature parameter significantly impacts output quality:
   - 0.5-0.7: Optimal for technical documentation and factual content
   - 0.7-0.8: Balanced for general purpose datasets
   - 0.8-1.0: Increased creativity, suitable for conversational data


   The system initiates the generation pipeline, leveraging Distilabel for structured output:
   ![image/png](https://cdn-uploads.huggingface.co/production/uploads/64461026e1fd8d65b27e6187/PWNT_bLHwFjeoFX7AhA-z.png)

   
   Upon completion, the dataset is pushed to Hugging Face Hub:
   ![Generation Complete](https://cdn-uploads.huggingface.co/production/uploads/64461026e1fd8d65b27e6187/ohd4S-RyNI406uLPf4bnZ.png)

   Access your generated dataset through the Hugging Face Hub interface:

<iframe
   src="https://huggingface.co/datasets/daqc/info-security-policies-rag-distiset/embed/viewer/default/train"
   frameborder="0"
   width="100%"
   height="560px"
></iframe>
   


## 3. Data Curation with Argilla

The integration with Argilla provides enterprise-grade dataset curation capabilities through a comprehensive review system. This phase is crucial for ensuring data quality and maintaining high standards in your training datasets.

### Environment Configuration
Before accessing Argilla's features, ensure proper configuration in your `.env` file.


### Curation Workflow

1. **Dataset Integration**
   Upon generation completion, the dataset is automatically ingested into Argilla. The system maintains data integrity and version control throughout the process. All datasets and progress persist across Docker restarts unless you explicitly remove the Argilla services and volumes.
   ![image/png](https://cdn-uploads.huggingface.co/production/uploads/64461026e1fd8d65b27e6187/0gF6iLywhKafEo3z94cd-.png)


2. **Quality Assurance Process**
   Argilla's interface provides comprehensive tools for dataset validation:
   - Semantic analysis of generated content
   - Consistency checking across entries
   - Metadata validation and enrichment
   - Collaborative review capabilities
   
   ![image/png](https://cdn-uploads.huggingface.co/production/uploads/64461026e1fd8d65b27e6187/h9kJ-4lA0LcFC8g6g_vwF.png)



3. **Dataset Publication**
   After thorough review, export your curated dataset to Hugging Face Hub:
   
   > Note: Consider using a new repository name to preserve both raw and curated datasets separately.
   
   - Configure repository settings
   - Set visibility and access controls
   - Add dataset cards and documentation
   
   ![Export Configuration](https://cdn-uploads.huggingface.co/production/uploads/64461026e1fd8d65b27e6187/CPwtVr_Jw6mndNCOU2a5T.png)


The curated dataset maintains full provenance tracking and quality metrics:
<iframe
   src="https://huggingface.co/datasets/daqc/info-security-policies-rag-distiset-argilla/embed/viewer/default/train"
   frameborder="0"
   width="100%"
   height="560px"
></iframe>

# 🎉 You're Done!
Congratulations! You've successfully completed the end-to-end dataset generation and curation process. Your curated dataset is now ready for model training.

## Experience the Solution

For a hands-on preview of the Synthetic Dataset Generator's capabilities, explore the hosted space. This allows you to evaluate the interface and functionality before deploying your own instance:

<iframe
  src="https://argilla-synthetic-data-generator.hf.space"
  frameborder="0"
  width="850"
  height="450"
  referrerpolicy="same-origin"
  sandbox="allow-scripts"
></iframe>

Create your own deployment by <a href="https://huggingface.co/spaces/argilla/synthetic-data-generator?duplicate=true">duplicating this Space</a>.

## What's Next?

After successfully generating your first dataset, several advanced implementation paths are available:

Extend your dataset generation capabilities:
- [Fine-tune models on synthetic data](https://huggingface.co/blog/davidberenstein1957/fine-tune-a-smollm-on-synthetic-data-of-llm) for domain-specific tasks
- Implement custom validation workflows
- [Create specialized reasoning datasets](https://huggingface.co/blog/sdiazlor/fine-tune-deepseek-with-a-synthetic-reasoning-data) for advanced model training
- Scale dataset generation for production needs

## Conclusion

The Synthetic Dataset Generator represents a significant advancement in private data generation technology, addressing the growing need for high-quality training data while maintaining security and control. By leveraging containerized architecture and local LLM inference, organizations can now generate custom datasets without compromising on data privacy or quality.

The solution's modular design enables seamless integration with existing ML pipelines while providing enterprise-grade features like persistent storage, comprehensive monitoring, and scalable infrastructure. Through collaborative validation workflows and structured quality control processes, teams can efficiently create and curate datasets tailored to their specific needs.

This combination of security, efficiency, and flexibility makes the Synthetic Dataset Generator an essential tool for organizations looking to accelerate their AI development while maintaining complete control over their data generation pipeline.

## References & Documentation


- [Synthetic Dataset Generator](https://github.com/argilla-io/synthetic-data-generator): Open-source tool for  dataset generation using natural language
- [Distilabel Framework](https://github.com/argilla-io/distilabel): Advanced dataset generation framework
- [Docker Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/): Container optimization guidelines
- [Argilla Documentation](https://docs.argilla.io): Data curation platform documentation
- [Ollama Integration](https://github.com/jmorganca/ollama): Local LLM deployment guide