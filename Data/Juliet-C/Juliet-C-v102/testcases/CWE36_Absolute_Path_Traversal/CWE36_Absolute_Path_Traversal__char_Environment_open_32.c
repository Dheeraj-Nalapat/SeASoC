/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE36_Absolute_Path_Traversal__char_Environment_open_32.c
Label Definition File: CWE36_Absolute_Path_Traversal.label.xml
Template File: sources-sink-32.tmpl.c
*/
/*
 * @description
 * CWE: 36 Absolute Path Traversal
 * BadSource: Environment Read input from an environment variable
 * GoodSource: Full path and file name
 * Sink: open
 *    BadSink :
 * Flow Variant: 32 Data flow using two pointers to the same value within the same function
 *
 * */

#include "std_testcase.h"

#define ENV_VARIABLE "ADD"

#ifdef _WIN32
# define GETENV getenv
#else
# define GETENV getenv
#endif

#ifdef _WIN32
# define OPEN _open
# define CLOSE _close
#else
# define OPEN open
# define CLOSE close
#endif

#ifndef OMITBAD

void CWE36_Absolute_Path_Traversal__char_Environment_open_32_bad()
{
    char * data;
    char * *data_ptr1 = &data;
    char * *data_ptr2 = &data;
    char data_buf[FILENAME_MAX] = "";
    data = data_buf;
    {
        char * data = *data_ptr1;
        {
            /* Read input from an environment variable */
            size_t data_len = strlen(data);
            char * environment = GETENV(ENV_VARIABLE);
            /* If there is data in the environment variable */
            if (environment != NULL)
            {
                strncat(data+data_len, environment, 100-data_len-1);
            }
        }
        *data_ptr1 = data;
    }
    {
        char * data = *data_ptr2;
        {
            int fd;
            /* POTENTIAL FLAW: Possibly opening a file without validating the file name or path */
            fd = OPEN(data, O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
            if (fd != -1)
            {
                CLOSE(fd);
            }
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    char * data;
    char * *data_ptr1 = &data;
    char * *data_ptr2 = &data;
    char data_buf[FILENAME_MAX] = "";
    data = data_buf;
    {
        char * data = *data_ptr1;
#ifdef _WIN32
        /* FIX: Full path and file name */
        strcpy(data, "c:\\temp\\file.txt");
#else
        /* FIX: Full path and file name */
        strcpy(data, "/tmp/file.txt");
#endif
        *data_ptr1 = data;
    }
    {
        char * data = *data_ptr2;
        {
            int fd;
            /* POTENTIAL FLAW: Possibly opening a file without validating the file name or path */
            fd = OPEN(data, O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
            if (fd != -1)
            {
                CLOSE(fd);
            }
        }
    }
}

void CWE36_Absolute_Path_Traversal__char_Environment_open_32_good()
{
    goodG2B();
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
    CWE36_Absolute_Path_Traversal__char_Environment_open_32_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE36_Absolute_Path_Traversal__char_Environment_open_32_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
