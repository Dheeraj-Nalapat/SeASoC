/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE400_Resource_Exhaustion__listen_socket_fwrite_61a.c
Label Definition File: CWE400_Resource_Exhaustion.label.xml
Template File: sources-sinks-61a.tmpl.c
*/
/*
 * @description
 * CWE: 400 Resource Exhaustion
 * BadSource: listen_socket Read data using a listen socket (server side)
 * GoodSource: Assign count to be a relatively small number
 * Sinks: fwrite
 *    GoodSink: Write to a file count number of times, but first validate count
 *    BadSink : Write to a file count number of times
 * Flow Variant: 61 Data flow: data returned from one function to another in different source files
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
#define CHAR_ARRAY_SIZE sizeof(count)*sizeof(count)

#define SENTENCE "This is the sentence we are printing to the file. "

#ifndef OMITBAD

/* bad function declaration */
int CWE400_Resource_Exhaustion__listen_socket_fwrite_61b_bad_source(int count);

void CWE400_Resource_Exhaustion__listen_socket_fwrite_61_bad()
{
    int count;
    /* Initialize count */
    count = -1;
    count = CWE400_Resource_Exhaustion__listen_socket_fwrite_61b_bad_source(count);
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

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
int CWE400_Resource_Exhaustion__listen_socket_fwrite_61b_goodG2B_source(int count);

static void goodG2B()
{
    int count;
    /* Initialize count */
    count = -1;
    count = CWE400_Resource_Exhaustion__listen_socket_fwrite_61b_goodG2B_source(count);
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

/* goodB2G uses the BadSource with the GoodSink */
int CWE400_Resource_Exhaustion__listen_socket_fwrite_61b_goodB2G_source(int count);

static void goodB2G()
{
    int count;
    /* Initialize count */
    count = -1;
    count = CWE400_Resource_Exhaustion__listen_socket_fwrite_61b_goodB2G_source(count);
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

void CWE400_Resource_Exhaustion__listen_socket_fwrite_61_good()
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
    CWE400_Resource_Exhaustion__listen_socket_fwrite_61_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE400_Resource_Exhaustion__listen_socket_fwrite_61_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
