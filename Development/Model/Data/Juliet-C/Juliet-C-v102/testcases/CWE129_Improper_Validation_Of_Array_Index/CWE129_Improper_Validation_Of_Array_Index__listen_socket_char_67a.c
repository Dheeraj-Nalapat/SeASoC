/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67a.c
Label Definition File: CWE129_Improper_Validation_Of_Array_Index.label.xml
Template File: sources-sinks-67a.tmpl.c
*/
/*
 * @description
 * CWE: 129 Improper Validation of Array Index
 * BadSource: listen_socket Read data using a listen socket (server side)
 * GoodSource: Larger than zero but less than 10
 * Sinks: char
 *    GoodSink: Ensure the array index is valid
 *    BadSink : Improperly check the array index by not checking the upper bound
 * Flow Variant: 67 Data flow: data passed in a struct from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# include <winsock2.h>
# include <windows.h>
# include <direct.h>
# pragma comment(lib, "ws2_32") /* include ws2_32.lib when linking */
# define CLOSE_SOCKET closesocket
#else
# define INVALID_SOCKET -1
# define SOCKET_ERROR -1
# define CLOSE_SOCKET close
# define SOCKET int
#endif

#define TCP_PORT 27015
#define LISTEN_BACKLOG 5
#define CHAR_ARRAY_SIZE sizeof(data)*sizeof(data)

typedef struct _CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67_struct_type
{
    int a;
} CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67_struct_type;

#ifndef OMITBAD

/* bad function declaration */
void CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67b_bad_sink(CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67_struct_type my_struct);

void CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67_bad()
{
    int data;
    CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67_struct_type my_struct;
    data = -1; /* Initialize data */
    {
#ifdef _WIN32
        WSADATA wsa_data;
        int wsa_data_init = 0;
#endif
        int recv_rv;
        struct sockaddr_in s_in;
        SOCKET listen_socket = INVALID_SOCKET;
        SOCKET accept_socket = INVALID_SOCKET;
        char input_buf[CHAR_ARRAY_SIZE];
        do
        {
#ifdef _WIN32
            if (WSAStartup(MAKEWORD(2,2), &wsa_data) != NO_ERROR) break;
            wsa_data_init = 1;
#endif
            listen_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if (listen_socket == INVALID_SOCKET) break;
            memset(&s_in, 0, sizeof(s_in));
            s_in.sin_family = AF_INET;
            s_in.sin_addr.s_addr = INADDR_ANY;
            s_in.sin_port = htons(TCP_PORT);
            if (bind(listen_socket, (struct sockaddr*)&s_in, sizeof(s_in)) == SOCKET_ERROR) break;
            if (listen(listen_socket, LISTEN_BACKLOG) == SOCKET_ERROR) break;
            accept_socket = accept(listen_socket, NULL, NULL);
            if (accept_socket == SOCKET_ERROR) break;
            /* Abort on error or the connection was closed */
            recv_rv = recv(accept_socket, input_buf, CHAR_ARRAY_SIZE, 0);
            if (recv_rv == SOCKET_ERROR || recv_rv == 0) break;
            /* Convert to int */
            data = atoi(input_buf);
        }
        while (0);
        if (listen_socket != INVALID_SOCKET) CLOSE_SOCKET(listen_socket);
        if (accept_socket != INVALID_SOCKET) CLOSE_SOCKET(accept_socket);
#ifdef _WIN32
        if (wsa_data_init) WSACleanup();
#endif
    }
    my_struct.a = data;
    CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67b_bad_sink(my_struct);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67b_goodG2B_sink(CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67_struct_type my_struct);

static void goodG2B()
{
    int data;
    CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67_struct_type my_struct;
    data = -1; /* Initialize data */
    /* FIX: Use a value greater than 0, but less than 10 to avoid attempting to
     * access an index of the array in the sink that is out-of-bounds */
    data = 7;
    my_struct.a = data;
    CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67b_goodG2B_sink(my_struct);
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67b_goodB2G_sink(CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67_struct_type my_struct);

static void goodB2G()
{
    int data;
    CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67_struct_type my_struct;
    data = -1; /* Initialize data */
    {
#ifdef _WIN32
        WSADATA wsa_data;
        int wsa_data_init = 0;
#endif
        int recv_rv;
        struct sockaddr_in s_in;
        SOCKET listen_socket = INVALID_SOCKET;
        SOCKET accept_socket = INVALID_SOCKET;
        char input_buf[CHAR_ARRAY_SIZE];
        do
        {
#ifdef _WIN32
            if (WSAStartup(MAKEWORD(2,2), &wsa_data) != NO_ERROR) break;
            wsa_data_init = 1;
#endif
            listen_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if (listen_socket == INVALID_SOCKET) break;
            memset(&s_in, 0, sizeof(s_in));
            s_in.sin_family = AF_INET;
            s_in.sin_addr.s_addr = INADDR_ANY;
            s_in.sin_port = htons(TCP_PORT);
            if (bind(listen_socket, (struct sockaddr*)&s_in, sizeof(s_in)) == SOCKET_ERROR) break;
            if (listen(listen_socket, LISTEN_BACKLOG) == SOCKET_ERROR) break;
            accept_socket = accept(listen_socket, NULL, NULL);
            if (accept_socket == SOCKET_ERROR) break;
            /* Abort on error or the connection was closed */
            recv_rv = recv(accept_socket, input_buf, CHAR_ARRAY_SIZE, 0);
            if (recv_rv == SOCKET_ERROR || recv_rv == 0) break;
            /* Convert to int */
            data = atoi(input_buf);
        }
        while (0);
        if (listen_socket != INVALID_SOCKET) CLOSE_SOCKET(listen_socket);
        if (accept_socket != INVALID_SOCKET) CLOSE_SOCKET(accept_socket);
#ifdef _WIN32
        if (wsa_data_init) WSACleanup();
#endif
    }
    my_struct.a = data;
    CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67b_goodB2G_sink(my_struct);
}

void CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67_good()
{
    goodG2B();
    goodB2G();
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
    CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE129_Improper_Validation_Of_Array_Index__listen_socket_char_67_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
