"""
Мини-RAG система с LangChain
Полное решение за 15 минут
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.document_loaders import TextLoader, UnstructuredMarkdownLoader
from qdrant_client import QdrantClient

@dataclass
class RAGConfig:
    """Конфигурация RAG системы"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "text-embedding-3-small"
    # URL of the running Qdrant instance (default assumes local server)
    vector_store_url: str = "http://localhost:6333"
    collection_name: str = "knowledge_base"
    search_k: int = 3
    # Ollama configuration
    llm_model: str = "mistral"  # 3B-7B GPT-like models: mistral, neural-chat, orca-mini:3b, phi
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.1
    generation_mode: str = "simple"  # "simple" (direct) or "chain" (RetrievalQA)

class MiniRAGSystem:
    """Полнофункциональная мини-RAG система"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        
        # Инициализация компонентов
        self._init_text_splitter()
        self._init_embeddings()
        self._init_vector_store()
        self._init_llm()
        self._init_prompt()
        
        # Статистика
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "queries_processed": 0
        }
    
    def _init_text_splitter(self):
        """Инициализация текстового сплиттера"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
    
    def _init_embeddings(self):
        """Инициализация модели эмбеддингов.

        Если конфигурация содержит ``text-embedding-3-small`` (или любой
        другой идентификатор, по которому можно загрузить модель через
        ``transformers``), используется собственный загрузчик из пакета
        ``src.embedding``.  Это позволяет запускать модель полностью
        локально без обращения к внешним сервисам.
        """
        # lazy import to avoid pulling the entire transformers stack when
        # the default ``HuggingFaceEmbeddings`` is sufficient.
        if self.config.embedding_model.startswith("text-embedding-3-small"):
            from src.embedding.transformers_embeddings import TransformersEmbeddings

            # ``TransformersEmbeddings`` implements ``__call__`` so it can be
            # passed directly to LangChain vector stores as the
            # ``embedding_function``.
            self.embeddings = TransformersEmbeddings(self.config.embedding_model)
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
    
    def _init_vector_store(self):
        """Инициализация векторного хранилища через Qdrant"""
        # подтягиваем qdrant-клиент (устанавливается из ``qdrant`` пакета)
        self.qdrant_client = QdrantClient(url=self.config.vector_store_url)

        # LangChain обёртка; она инкапсулирует ``client`` и обеспечивает
        # единый интерфейс для хранилища.  По умолчанию она создаёт
        # коллекцию при необходимости.
        self.vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=self.config.collection_name,
            embeddings=self.embeddings,
            prefer_grpc=True,
        )
    
    def _init_llm(self):
        """Инициализация LLM с Ollama (локальная self-hosted модель)"""
        try:
            # Используем Ollama для локальной self-hosted модели
            self.llm = Ollama(
                base_url=self.config.ollama_base_url,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                top_k=40,
                top_p=0.9,
            )
            # Проверяем доступность модели
            print(f"✅ Ollama подключена: {self.config.llm_model} @ {self.config.ollama_base_url}")
        except Exception as e:
            print(f"❌ Ошибка подключения к Ollama: {e}")
            print("⚠️  Пожалуйста убедитесь что Ollama запущена:")
            print("    macOS/Linux: ollama serve")
            print("    Windows: запустите приложение Ollama")
            print("    Скачать: https://ollama.ai")
            print(f"    Загрузить модель: ollama pull {self.config.llm_model}")
            raise
    
    def _init_prompt(self):
        """Инициализация промпт-шаблона"""
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Ты - полезный ассистент по технической документации.
Используй ТОЛЬКО предоставленный контекст для ответа.
Если ответа нет в контексте, скажи "Информация не найдена в документации".

Контекст:
{context}

Вопрос: {question}

Ответ (будь краток и точен):
"""
        )

    def _generate_answer_simple(self, context: str, question: str) -> str:
        """
        Простой и легкий метод генерации ответа без RetrievalQA цепочки
        Прямой вызов LLM с отформатированным промптом
        """
        # Форматируем промпт преимущественно
        prompt = self.prompt_template.format(context=context, question=question)

        # Вызываем LLM напрямую
        try:
            # Для ChatOpenAI
            if hasattr(self.llm, 'predict'):
                answer = self.llm.predict(prompt)
            # Для HuggingFacePipeline и других моделей
            else:
                from langchain.schema import HumanMessage
                messages = [HumanMessage(content=prompt)]
                response = self.llm.generate(messages)
                answer = response.generations[0][0].text
        except Exception as e:
            # Как fallback используем invoke если доступен
            try:
                response = self.llm.invoke(prompt)
                if hasattr(response, 'content'):
                    answer = response.content
                else:
                    answer = str(response)
            except Exception as e2:
                answer = f"Ошибка при генерировании ответа: {str(e2)}"

        return answer.strip() if answer else "Не удалось сгенерировать ответ"

    
    def add_document(self, text: str, metadata: Optional[Dict] = None) -> List[str]:
        """
        Добавить документ в систему
        Возвращает список ID чанков
        """
        # Базовые метаданные
        base_metadata = {
            "source": "manual_input",
            "timestamp": datetime.now().isoformat(),
            "doc_hash": hashlib.md5(text.encode()).hexdigest()[:8]
        }
        
        # Объединяем с пользовательскими метаданными
        if metadata:
            base_metadata.update(metadata)
        
        # Создаем документ LangChain
        document = Document(
            page_content=text,
            metadata=base_metadata
        )
        
        # Разбиваем на чанки
        chunks = self.text_splitter.split_documents([document])
        
        # Обогащаем метаданные чанков
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{base_metadata['doc_hash']}_chunk_{i}"
            chunk.metadata.update({
                "chunk_id": chunk_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "parent_doc_hash": base_metadata['doc_hash']
            })
            chunk_ids.append(chunk_id)
        
        # Добавляем в векторное хранилище
        self.vector_store.add_documents(chunks)
        
        # Обновляем статистику
        self.stats["documents_processed"] += 1
        self.stats["chunks_created"] += len(chunks)
        
        # Персистируем изменения
        self.vector_store.persist()
        
        return chunk_ids
    
    def add_document_from_file(self, file_path: str, file_type: str = "auto") -> List[str]:
        """Добавить документ из файла"""
        # Определяем загрузчик по типу файла
        if file_type == "auto":
            if file_path.endswith(".md"):
                file_type = "markdown"
            elif file_path.endswith(".txt"):
                file_type = "text"
        
        if file_type == "markdown":
            loader = UnstructuredMarkdownLoader(file_path)
        else:  # text
            loader = TextLoader(file_path)
        
        # Загружаем документ
        documents = loader.load()
        
        # Обрабатываем каждый документ
        chunk_ids = []
        for doc in documents:
            doc.metadata["source"] = file_path
            chunk_ids.extend(self.add_document(doc.page_content, doc.metadata))
        
        return chunk_ids
    
    def query(self, question: str, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Поиск ответа на вопрос
        Возвращает полную информацию о результатах
        """
        search_k = k or self.config.search_k

        # 1. Поиск релевантных чанков
        relevant_docs = self.vector_store.similarity_search_with_score(
            question,
            k=search_k
        )

        # 2. Форматирование контекста
        context_parts = []
        sources = []

        for i, (doc, score) in enumerate(relevant_docs):
            context_parts.append(
                f"[Документ {i+1}, релевантность: {score:.3f}]:\n{doc.page_content}"
            )
            sources.append({
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata,
                "score": float(score)
            })

        context = "\n\n".join(context_parts)

        # 3. Генерация ответа
        if self.config.generation_mode == "simple":
            # Используем простой и легкий метод генерации
            answer = self._generate_answer_simple(context, question)
        else:
            # Используем RetrievalQA цепочку (исходный метод)
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": search_k}
                ),
                chain_type_kwargs={"prompt": self.prompt_template},
                return_source_documents=True
            )

            result = qa_chain({"query": question})
            answer = result["result"]

        # 4. Форматирование ответа
        response = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "generation_mode": self.config.generation_mode,
            "stats": {
                "docs_retrieved": len(relevant_docs),
                "avg_relevance_score": (
                    sum(score for _, score in relevant_docs) / len(relevant_docs)
                    if relevant_docs else 0
                ),
                "context_length": len(context)
            },
            "raw_context": context if len(context) < 1000 else context[:1000] + "..."
        }

        # Обновляем статистику
        self.stats["queries_processed"] += 1

        return response
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        """Поиск похожих документов без генерации ответа"""
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content[:500],
                "metadata": doc.metadata,
                "relevance_score": float(score),
                "source": doc.metadata.get("source", "unknown")
            })
        
        return formatted_results
    
    def delete_document(self, doc_hash: str) -> bool:
        """Удалить документ по хэшу"""
        try:
            # Получаем все чанки документа
            collection = self.vector_store._collection
            results = collection.get(
                where={"parent_doc_hash": doc_hash},
                include=["metadatas", "documents"]
            )
            
            if results["ids"]:
                # Удаляем чанки
                collection.delete(ids=results["ids"])
                self.vector_store.persist()
                print(f"✅ Удалено {len(results['ids'])} чанков документа {doc_hash}")
                return True
            else:
                print(f"⚠️  Документ {doc_hash} не найден")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка при удалении: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Получить статистику системы"""
        # Получаем количество записей из Qdrant напрямую
        try:
            count = self.qdrant_client.count(collection_name=self.config.collection_name).count
        except Exception:
            count = 0

        stats = self.stats.copy()
        stats.update({
            "total_chunks_in_store": count,
            "embedding_model": self.config.embedding_model,
            "chunk_size": self.config.chunk_size,
            "vector_store_url": self.config.vector_store_url,
        })

        return stats
    
    def export_config(self) -> Dict:
        """Экспортировать конфигурацию системы"""
        return {
            "config": self.config.__dict__,
            "stats": self.get_stats(),
            "components": {
                "embeddings": self.config.embedding_model,
                "vector_store": "Qdrant",
                "llm": self.config.llm_model,
                "text_splitter": "RecursiveCharacterTextSplitter",
            },
        }

# ==================== ДЕМОНСТРАЦИЯ РАБОТЫ ====================

def demo_rag_system():
    """Демонстрация работы RAG системы"""
    print("🚀 Запуск демонстрации Mini-RAG системы...\n")

    # 1. Инициализация системы
    config = RAGConfig(
        chunk_size=800,
        chunk_overlap=150,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="mistral",  # Самохостед модель через Ollama
        ollama_base_url="http://localhost:11434"
    )

    rag = MiniRAGSystem(config)
    print("✅ RAG система инициализирована")
    print(f"📊 Конфигурация: чанки {config.chunk_size}/{config.chunk_overlap}")
    print(f"🤖 Модель: {config.embedding_model} + Ollama ({config.llm_model})\n")
    
    # 2. Добавление тестовых документов
    test_docs = [
        """
        # Деплой микросервисов
        
        Мы используем Kubernetes для деплоя микросервисов.
        Каждый сервис упакован в Docker контейнер.
        
        Процесс деплоя:
        1. Сборка образа Docker
        2. Пуш в Container Registry
        3. Обновление манифеста Kubernetes
        4. Rolling update через Helm
        
        Мониторинг осуществляется через Prometheus и Grafana.
        Логи собираются в ELK стек.
        """,
        
        """
        # Onboarding новых сотрудников
        
        Шаги для нового сотрудника:
        1. Получить доступ к корпоративным системам
        2. Настроить локальное окружение (Docker, IDE)
        3. Изучить документацию по архитектуре
        4. Присоединиться к командам в Slack
        
        Важные ссылки:
        - Wiki: https://wiki.company.com
        - GitLab: https://gitlab.company.com
        - Jenkins: https://jenkins.company.com
        
        Контакты:
        - Ментор: Иван Иванов (ivan@company.com)
        - HR: hr@company.com
        """,
        
        """
        # Аутентификация и авторизация
        
        Мы используем OAuth 2.0 для аутентификации.
        Все сервисы должны использовать единый Identity Provider.
        
        Роли пользователей:
        - ADMIN: полный доступ ко всем системам
        - DEVELOPER: доступ к разработке и тестированию
        - VIEWER: только чтение
        
        Токены JWT живут 24 часа.
        Refresh tokens живут 30 дней.
        
        Важно: никогда не храните секреты в коде!
        Используйте HashiCorp Vault для управления секретами.
        """
    ]
    
    print("📄 Добавление тестовых документов...")
    for i, doc_text in enumerate(test_docs):
        chunk_ids = rag.add_document(
            doc_text,
            metadata={
                "title": f"Test Doc {i+1}",
                "category": "technical",
                "author": "system"
            }
        )
        print(f"  Документ {i+1}: создано {len(chunk_ids)} чанков")
    
    print(f"\n📊 Статистика после добавления:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 3. Выполнение запросов
    print("\n🔍 Выполнение тестовых запросов...\n")

    test_queries = [
        "Как мы деплоим микросервисы?",
        "Что нужно сделать новому сотруднику?",
        "Как работает аутентификация в нашей системе?",
        "Какие инструменты мониторинга мы используем?"
    ]

    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ПРОСТОГО МЕТОДА ГЕНЕРАЦИИ (small LLM)")
    print("=" * 60)
    for query in test_queries[:2]:  # Первые два запроса через простой метод
        print(f"\n❓ Вопрос: {query}")
        result = rag.query(query, k=2)

        print(f"💡 Ответ: {result['answer'][:150]}...")
        print(f"📈 Релевантность: {result['stats']['avg_relevance_score']:.3f}")
        print(f"⚙️  Режим генерации: {result['generation_mode']}")

        if result['sources']:
            print(f"📚 Источники:")
            for i, source in enumerate(result['sources'][:2]):
                print(f"  {i+1}. [{source['metadata'].get('title', 'No title')}] "
                      f"(score: {source['score']:.3f})")

    # Переключаемся на цепочку для сравнения
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ МЕТОДА С ЦЕПОЧКОЙ (RetrievalQA)")
    print("=" * 60)
    rag.config.generation_mode = "chain"

    for query in test_queries[2:]:  # Следующие два запроса через цепочку
        print(f"\n❓ Вопрос: {query}")
        result = rag.query(query, k=2)

        print(f"💡 Ответ: {result['answer'][:150]}...")
        print(f"📈 Релевантность: {result['stats']['avg_relevance_score']:.3f}")
        print(f"⚙️  Режим генерации: {result['generation_mode']}")

        if result['sources']:
            print(f"📚 Источники:")
            for i, source in enumerate(result['sources'][:2]):
                print(f"  {i+1}. [{source['metadata'].get('title', 'No title')}] "
                      f"(score: {source['score']:.3f})")
    
    # 4. Поиск похожих документов
    print("🔎 Поиск похожих документов...")
    similar = rag.search_similar("как настроить окружение для разработки", k=2)
    for i, doc in enumerate(similar):
        print(f"  {i+1}. {doc['content'][:100]}... (score: {doc['relevance_score']:.3f})")
    
    # 5. Экспорт конфигурации
    print("\n💾 Экспорт конфигурации системы...")
    config_export = rag.export_config()
    print(f"  Модель эмбеддингов: {config_export['components']['embeddings']}")
    print(f"  Всего чанков: {config_export['stats']['total_chunks_in_store']}")
    print(f"  Обработано запросов: {config_export['stats']['queries_processed']}")
    
    print("\n🎉 Демонстрация завершена успешно!")
    return rag

# ==================== ТЕСТИРОВАНИЕ ====================

def test_edge_cases(rag: MiniRAGSystem):
    """Тестирование edge cases"""
    print("\n🧪 Тестирование edge cases...")
    
    # 1. Пустой запрос
    print("1. Пустой запрос:")
    result = rag.query("")
    print(f"   Ответ: {result['answer'][:50]}...")
    
    # 2. Запрос без ответа в документах
    print("\n2. Запрос без ответа в документах:")
    result = rag.query("Как приготовить пиццу?")
    print(f"   Ответ: {result['answer']}")
    
    # 3. Очень длинный запрос
    print("\n3. Длинный запрос:")
    long_query = "Как настроить " + " и ".join([f"систему{i}" for i in range(10)])
    result = rag.query(long_query)
    print(f"   Найдено документов: {result['stats']['docs_retrieved']}")
    
    # 4. Удаление документа
    print("\n4. Удаление документа:")
    # Сначала найдем хэш документа
    test_query = "деплой микросервисов"
    similar = rag.search_similar(test_query, k=1)
    if similar:
        doc_hash = similar[0]['metadata'].get('parent_doc_hash')
        if doc_hash:
            success = rag.delete_document(doc_hash)
            print(f"   Удаление {'успешно' if success else 'неудачно'}")
    
    print("✅ Тестирование edge cases завершено")

# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    print("=" * 60)
    print("MINI-RAG SYSTEM WITH LANGCHAIN")
    print("Полное решение для 30-минутного интервью")
    print("=" * 60)
    
    # Демонстрация
    rag_system = demo_rag_system()
    
    # Тестирование edge cases (опционально)
    # test_edge_cases(rag_system)
    
    # Пример использования извне
    print("\n📋 Пример использования в production:")
    print("""
    # Инициализация с self-hosted моделью через Ollama
    from langchain.llms import Ollama

    config = RAGConfig(
        llm_model="mistral",  # или другая модель
        ollama_base_url="http://localhost:11434"
    )
    rag = MiniRAGSystem(config)

    # Добавление документов
    rag.add_document("Ваш текст документа", metadata={"source": "api"})

    # Поиск ответа
    result = rag.query("Ваш вопрос")
    print(result['answer'])

    # Мониторинг
    stats = rag.get_stats()
    print(f"Обработано запросов: {stats['queries_processed']}")

    # Доступные модели в Ollama (3B-7B GPT-like):
    # - mistral (7B, очень быстро и хорошо)
    # - neural-chat (7B)
    # - orca-mini:3b (3B, компактно)
    # - phi (2.7B, очень быстро)
    """)