#ifndef THREADING
#define THREADING

#include "threading.h"
#include<iostream>

namespace threading {

            /**
             * Exception type
             */
            enum ExceptionType {
                /**
                 * Attaching current thread to JVM failed.
                 */
                ATTACH_FAILED,
                /**
                 * Detaching current thread from JVM failed.
                 */
                DETACH_FAILED,
                /**
                 * Current thread not attached to JVM.
                 */
                NOT_ATTACHED
            };

            /**
             * Exception
             */
            class JNIThreadException {
            public:

                /**
                 * Constructor.
                 * @param type exception type
                 */
                JNIThreadException(ExceptionType type) {
                    this->type = type;

                    switch (type) {
                        case ATTACH_FAILED:
                        {
                            std::cerr << "Attaching thread failed!" << std::endl;

                        }
                            break;
                        case DETACH_FAILED:
                        {
                            std::cerr << "Detaching thread failed!" << std::endl;
                        }
                            break;
                        case NOT_ATTACHED:
                        {
                        std::cerr << "Thread not attached!" << std::endl;
                        }
                            break;
                    }
                }

                ExceptionType type;
            };

            /**
             * Attaches the current thread to the JVM. If the thread is already
             * attached this is equivalent to <code>getEnv()</code>.
             * @param javaVM Java VM to operate on
             * @return JVM environment of the current thread
             * @throws JNIThreadException
             */
            inline JNIEnv* attachThread(JavaVM* javaVM) {

                // The following code raised a warning in newer GCC versions:
                // "dereferencing type-punned pointer will break strict-aliasing rules"
                // That is why we do it differently now, although this code
                // is officially used:
                //				JNIEnv* localEnv = NULL;
                //
                //				int result = javaVM->AttachCurrentThread(
                //						(void **) (&localEnv), NULL);

                JNIEnv** localEnvPtrPtr;
                JNIEnv* localEnv = NULL;
                localEnvPtrPtr = &localEnv;

                int result = javaVM->AttachCurrentThread(
                        (void **) (localEnvPtrPtr), NULL);

                if (result < 0) {
                    throw JNIThreadException(ATTACH_FAILED);
                }

                return localEnv;
            }

            /**
             * Detaches the current thread from the JVM.
             * @param javaVM Java VM to operate on
             * @throws JNIThreadException
             */
            inline void detachThread(JavaVM* javaVM) {

                int result = javaVM->DetachCurrentThread();

                if (result < 0) {
                    throw JNIThreadException(DETACH_FAILED);
                }
            }

            /**
             * Returns the JVM environment of the current thread.
             * @param javaVM Java VM to operate on
             * @return JVM environment of the current thread
             * @throws JNIThreadException
             */
            inline JNIEnv* getEnv(JavaVM* javaVM) {

                // The following code raised a warning in newer GCC versions:
                // "dereferencing type-punned pointer will break strict-aliasing rules"
                // That is why we do it differently now, although this code
                // is officially used:
                //				JNIEnv* localEnv = NULL;
                //
                //				jint result = javaVM->GetEnv(
                //						(void **) (&localEnv), JNI_VERSION_1_2);

                JNIEnv** localEnvPtrPtr;
                JNIEnv* localEnv = NULL;
                localEnvPtrPtr = &localEnv;

                jint result = javaVM->GetEnv(
                        (void **) (localEnvPtrPtr), JNI_VERSION_1_2);

                if (result != JNI_OK) {
                    throw JNIThreadException(NOT_ATTACHED);
                }

                return localEnv;
            }
        } // threading::

#endif // THREADING

