/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE400_Resource_Exhaustion__connect_socket_fwrite_11.c
Label Definition File: CWE400_Resource_Exhaustion.label.xml
Template File: sources-sinks-11.tmpl.c
*/
/*
 * @description
 * CWE: 400 Resource Exhaustion
 * BadSource: connect_socket Read data using a connect socket (client side)
 * GoodSource: Assign count to be a relatively small number
 * Sinks: fwrite
 *    GoodSink: Write to a file count number of times, but first validate count
 *    BadSink : Write to a file count number of times
 * Flow Variant: 11 Control flow: if(global_returns_t()) and if(global_returns_f())
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# include <winsock2.h>
# include <windows.h>
# include <direct.h>
# pragma comment(lib, "ws2_32") /* include ws2_32.lib when linking */
# define CLOSE_SOCKET closesocket
#else /* NOT _WIN32 */
# define INVALID_SOCKET -1
# define SOCKET_ERROR -1
# define CLOSE_SOCKET close
# define SOCKET int
#endif

#define TCP_PORT 27015
#define CHAR_ARRAY_SIZE sizeof(count)*sizeof(count)

#define SENTENCE "This is the sentence we are printing to the file. "

#ifndef OMITBAD

void CWE400_Resource_Exhaustion__connect_socket_fwrite_11_bad()
{
    int count;
    /* Initialize count */
    count = -1;
    if(global_returns_t())
    {
        {
#ifdef _WIN32
            WSADATA wsa_data;
            int wsa_data_init = 0;
#endif
            int recv_rv;
            struct sockaddr_in s_in;
            SOCKET connect_socket = INVALID_SOCKET;
            char input_buf[CHAR_ARRAY_SIZE];
            do
            {
#ifdef _WIN32
                if (WSAStartup(MAKEWORD(2,2), &wsa_data) != NO_ERROR) break;
                wsa_data_init = 1;
#endif
                connect_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
                if (connect_socket == INVALID_SOCKET) break;
                memset(&s_in, 0, sizeof(s_in));
                s_in.sin_family = AF_INET;
                s_in.sin_addr.s_addr = inet_addr("127.0.0.1");
                s_in.sin_port = htons(TCP_PORT);
                if (connect(connect_socket, (struct sockaddr*)&s_in, sizeof(s_in)) == SOCKET_ERROR) break;
                /* Abort on error or the connection was closed, make sure to recv one
                 * less char than is in the recv_buf in order to append a terminator */
                recv_rv = recv(connect_socket, input_buf, CHAR_ARRAY_SIZE, 0);
                if (recv_rv == SOCKET_ERROR || recv_rv == 0) break;
                /* Convert to int */
                count = atoi(input_buf);
            }
            while (0);
            if (connect_socket != INVALID_SOCKET) CLOSE_SOCKET(connect_socket);
#ifdef _WIN32
            if (wsa_data_init) WSACleanup();
#endif
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use a relatively small number */
        count = 20;
    }
    if(global_returns_t())
    {
        {
            size_t i = 0;
            FILE *file = NULL;
            const char *filename = "output_bad.txt";
            file = fopen(filename, "w+");
            if (file == NULL) exit(1);
            /* FLAW: For loop using count as the loop variant and no validation
             * This can cause a file to become very large */
            for (i = 0; i < (size_t)count; i++)
            {
                if (strlen(SENTENCE) != fwrite(SENTENCE, sizeof(char), strlen(SENTENCE), file)) exit(1);
            }
            if (file)
            {
                fclose(file);
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            size_t i = 0;
            FILE *file = NULL;
            const char *filename = "output_good.txt";
            /* FIX: Validate count before using it as the for loop variant to write to a file */
            if (count > 0 && count <= 20)
            {
                file = fopen(filename, "w+");
                if (file == NULL) exit(1);
                for (i = 0; i < (size_t)count; i++)
                {
                    if (strlen(SENTENCE) != fwrite(SENTENCE, sizeof(char), strlen(SENTENCE), file)) exit(1);
                }
                if (file)
                {
                    fclose(file);
                }
            }
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G1() - use badsource and goodsink by changing the second global_returns_t() to global_returns_f() */
static void goodB2G1()
{
    int count;
    /* Initialize count */
    count = -1;
    if(global_returns_t())
    {
        {
#ifdef _WIN32
            WSADATA wsa_data;
            int wsa_data_init = 0;
#endif
            int recv_rv;
            struct sockaddr_in s_in;
            SOCKET connect_socket = INVALID_SOCKET;
            char input_buf[CHAR_ARRAY_SIZE];
            do
            {
#ifdef _WIN32
                if (WSAStartup(MAKEWORD(2,2), &wsa_data) != NO_ERROR) break;
                wsa_data_init = 1;
#endif
                connect_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
                if (connect_socket == INVALID_SOCKET) break;
                memset(&s_in, 0, sizeof(s_in));
                s_in.sin_family = AF_INET;
                s_in.sin_addr.s_addr = inet_addr("127.0.0.1");
                s_in.sin_port = htons(TCP_PORT);
                if (connect(connect_socket, (struct sockaddr*)&s_in, sizeof(s_in)) == SOCKET_ERROR) break;
                /* Abort on error or the connection was closed, make sure to recv one
                 * less char than is in the recv_buf in order to append a terminator */
                recv_rv = recv(connect_socket, input_buf, CHAR_ARRAY_SIZE, 0);
                if (recv_rv == SOCKET_ERROR || recv_rv == 0) break;
                /* Convert to int */
                count = atoi(input_buf);
            }
            while (0);
            if (connect_socket != INVALID_SOCKET) CLOSE_SOCKET(connect_socket);
#ifdef _WIN32
            if (wsa_data_init) WSACleanup();
#endif
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use a relatively small number */
        count = 20;
    }
    if(global_returns_f())
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            size_t i = 0;
            FILE *file = NULL;
            const char *filename = "output_bad.txt";
            file = fopen(filename, "w+");
            if (file == NULL) exit(1);
            /* FLAW: For loop using count as the loop variant and no validation
             * This can cause a file to become very large */
            for (i = 0; i < (size_t)count; i++)
            {
                if (strlen(SENTENCE) != fwrite(SENTENCE, sizeof(char), strlen(SENTENCE), file)) exit(1);
            }
            if (file)
            {
                fclose(file);
            }
        }
    }
    else
    {
        {
            size_t i = 0;
            FILE *file = NULL;
            const char *filename = "output_good.txt";
            /* FIX: Validate count before using it as the for loop variant to write to a file */
            if (count > 0 && count <= 20)
            {
                file = fopen(filename, "w+");
                if (file == NULL) exit(1);
                for (i = 0; i < (size_t)count; i++)
                {
                    if (strlen(SENTENCE) != fwrite(SENTENCE, sizeof(char), strlen(SENTENCE), file)) exit(1);
                }
                if (file)
                {
                    fclose(file);
                }
            }
        }
    }
}

/* goodB2G2() - use badsource and goodsink by reversing the blocks in the second if */
static void goodB2G2()
{
    int count;
    /* Initialize count */
    count = -1;
    if(global_returns_t())
    {
        {
#ifdef _WIN32
            WSADATA wsa_data;
            int wsa_data_init = 0;
#endif
            int recv_rv;
            struct sockaddr_in s_in;
            SOCKET connect_socket = INVALID_SOCKET;
            char input_buf[CHAR_ARRAY_SIZE];
            do
            {
#ifdef _WIN32
                if (WSAStartup(MAKEWORD(2,2), &wsa_data) != NO_ERROR) break;
                wsa_data_init = 1;
#endif
                connect_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
                if (connect_socket == INVALID_SOCKET) break;
                memset(&s_in, 0, sizeof(s_in));
                s_in.sin_family = AF_INET;
                s_in.sin_addr.s_addr = inet_addr("127.0.0.1");
                s_in.sin_port = htons(TCP_PORT);
                if (connect(connect_socket, (struct sockaddr*)&s_in, sizeof(s_in)) == SOCKET_ERROR) break;
                /* Abort on error or the connection was closed, make sure to recv one
                 * less char than is in the recv_buf in order to append a terminator */
                recv_rv = recv(connect_socket, input_buf, CHAR_ARRAY_SIZE, 0);
                if (recv_rv == SOCKET_ERROR || recv_rv == 0) break;
                /* Convert to int */
                count = atoi(input_buf);
            }
            while (0);
            if (connect_socket != INVALID_SOCKET) CLOSE_SOCKET(connect_socket);
#ifdef _WIN32
            if (wsa_data_init) WSACleanup();
#endif
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Use a relatively small number */
        count = 20;
    }
    if(global_returns_t())
    {
        {
            size_t i = 0;
            FILE *file = NULL;
            const char *filename = "output_good.txt";
            /* FIX: Validate count before using it as the for loop variant to write to a file */
            if (count > 0 && count <= 20)
            {
                file = fopen(filename, "w+");
                if (file == NULL) exit(1);
                for (i = 0; i < (size_t)count; i++)
                {
                    if (strlen(SENTENCE) != fwrite(SENTENCE, sizeof(char), strlen(SENTENCE), file)) exit(1);
                }
                if (file)
                {
                    fclose(file);
                }
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            size_t i = 0;
            FILE *file = NULL;
            const char *filename = "output_bad.txt";
            file = fopen(filename, "w+");
            if (file == NULL) exit(1);
            /* FLAW: For loop using count as the loop variant and no validation
             * This can cause a file to become very large */
            for (i = 0; i < (size_t)count; i++)
            {
                if (strlen(SENTENCE) != fwrite(SENTENCE, sizeof(char), strlen(SENTENCE), file)) exit(1);
            }
            if (file)
            {
                fclose(file);
            }
        }
    }
}

/* goodG2B1() - use goodsource and badsink by changing the first global_returns_t() to global_returns_f() */
static void goodG2B1()
{
    int count;
    /* Initialize count */
    count = -1;
    if(global_returns_f())
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
#ifdef _WIN32
            WSADATA wsa_data;
            int wsa_data_init = 0;
#endif
            int recv_rv;
            struct sockaddr_in s_in;
            SOCKET connect_socket = INVALID_SOCKET;
            char input_buf[CHAR_ARRAY_SIZE];
            do
            {
#ifdef _WIN32
                if (WSAStartup(MAKEWORD(2,2), &wsa_data) != NO_ERROR) break;
                wsa_data_init = 1;
#endif
                connect_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
                if (connect_socket == INVALID_SOCKET) break;
                memset(&s_in, 0, sizeof(s_in));
                s_in.sin_family = AF_INET;
                s_in.sin_addr.s_addr = inet_addr("127.0.0.1");
                s_in.sin_port = htons(TCP_PORT);
                if (connect(connect_socket, (struct sockaddr*)&s_in, sizeof(s_in)) == SOCKET_ERROR) break;
                /* Abort on error or the connection was closed, make sure to recv one
                 * less char than is in the recv_buf in order to append a terminator */
                recv_rv = recv(connect_socket, input_buf, CHAR_ARRAY_SIZE, 0);
                if (recv_rv == SOCKET_ERROR || recv_rv == 0) break;
                /* Convert to int */
                count = atoi(input_buf);
            }
            while (0);
            if (connect_socket != INVALID_SOCKET) CLOSE_SOCKET(connect_socket);
#ifdef _WIN32
            if (wsa_data_init) WSACleanup();
#endif
        }
    }
    else
    {
        /* FIX: Use a relatively small number */
        count = 20;
    }
    if(global_returns_t())
    {
        {
            size_t i = 0;
            FILE *file = NULL;
            const char *filename = "output_bad.txt";
            file = fopen(filename, "w+");
            if (file == NULL) exit(1);
            /* FLAW: For loop using count as the loop variant and no validation
             * This can cause a file to become very large */
            for (i = 0; i < (size_t)count; i++)
            {
                if (strlen(SENTENCE) != fwrite(SENTENCE, sizeof(char), strlen(SENTENCE), file)) exit(1);
            }
            if (file)
            {
                fclose(file);
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            size_t i = 0;
            FILE *file = NULL;
            const char *filename = "output_good.txt";
            /* FIX: Validate count before using it as the for loop variant to write to a file */
            if (count > 0 && count <= 20)
            {
                file = fopen(filename, "w+");
                if (file == NULL) exit(1);
                for (i = 0; i < (size_t)count; i++)
                {
                    if (strlen(SENTENCE) != fwrite(SENTENCE, sizeof(char), strlen(SENTENCE), file)) exit(1);
                }
                if (file)
                {
                    fclose(file);
                }
            }
        }
    }
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the first if */
static void goodG2B2()
{
    int count;
    /* Initialize count */
    count = -1;
    if(global_returns_t())
    {
        /* FIX: Use a relatively small number */
        count = 20;
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
#ifdef _WIN32
            WSADATA wsa_data;
            int wsa_data_init = 0;
#endif
            int recv_rv;
            struct sockaddr_in s_in;
            SOCKET connect_socket = INVALID_SOCKET;
            char input_buf[CHAR_ARRAY_SIZE];
            do
            {
#ifdef _WIN32
                if (WSAStartup(MAKEWORD(2,2), &wsa_data) != NO_ERROR) break;
                wsa_data_init = 1;
#endif
                connect_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
                if (connect_socket == INVALID_SOCKET) break;
                memset(&s_in, 0, sizeof(s_in));
                s_in.sin_family = AF_INET;
                s_in.sin_addr.s_addr = inet_addr("127.0.0.1");
                s_in.sin_port = htons(TCP_PORT);
                if (connect(connect_socket, (struct sockaddr*)&s_in, sizeof(s_in)) == SOCKET_ERROR) break;
                /* Abort on error or the connection was closed, make sure to recv one
                 * less char than is in the recv_buf in order to append a terminator */
                recv_rv = recv(connect_socket, input_buf, CHAR_ARRAY_SIZE, 0);
                if (recv_rv == SOCKET_ERROR || recv_rv == 0) break;
                /* Convert to int */
                count = atoi(input_buf);
            }
            while (0);
            if (connect_socket != INVALID_SOCKET) CLOSE_SOCKET(connect_socket);
#ifdef _WIN32
            if (wsa_data_init) WSACleanup();
#endif
        }
    }
    if(global_returns_t())
    {
        {
            size_t i = 0;
            FILE *file = NULL;
            const char *filename = "output_bad.txt";
            file = fopen(filename, "w+");
            if (file == NULL) exit(1);
            /* FLAW: For loop using count as the loop variant and no validation
             * This can cause a file to become very large */
            for (i = 0; i < (size_t)count; i++)
            {
                if (strlen(SENTENCE) != fwrite(SENTENCE, sizeof(char), strlen(SENTENCE), file)) exit(1);
            }
            if (file)
            {
                fclose(file);
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            size_t i = 0;
            FILE *file = NULL;
            const char *filename = "output_good.txt";
            /* FIX: Validate count before using it as the for loop variant to write to a file */
            if (count > 0 && count <= 20)
            {
                file = fopen(filename, "w+");
                if (file == NULL) exit(1);
                for (i = 0; i < (size_t)count; i++)
                {
                    if (strlen(SENTENCE) != fwrite(SENTENCE, sizeof(char), strlen(SENTENCE), file)) exit(1);
                }
                if (file)
                {
                    fclose(file);
                }
            }
        }
    }
}

void CWE400_Resource_Exhaustion__connect_socket_fwrite_11_good()
{
    goodB2G1();
    goodB2G2();
    goodG2B1();
    goodG2B2();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE400_Resource_Exhaustion__connect_socket_fwrite_11_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE400_Resource_Exhaustion__connect_socket_fwrite_11_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
