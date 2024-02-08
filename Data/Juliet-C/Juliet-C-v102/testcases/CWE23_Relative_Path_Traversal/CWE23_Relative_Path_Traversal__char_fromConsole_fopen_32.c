/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE23_Relative_Path_Traversal__char_fromConsole_fopen_32.c
Label Definition File: CWE23_Relative_Path_Traversal.label.xml
Template File: sources-sink-32.tmpl.c
*/
/*
 * @description
 * CWE: 23 Relative Path Traversal
 * BadSource: fromConsole Read input from the console
 * GoodSource: File name without a period or slash
 * Sink: fopen
 *    BadSink :
 * Flow Variant: 32 Data flow using two pointers to the same value within the same function
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
#define BASEPATH "c:\\temp\\"
#else
#define BASEPATH "/tmp/"
#endif

#ifdef _WIN32
# define FOPEN fopen
#else
/* fopen is used on unix-based OSs */
# define FOPEN fopen
#endif

#ifndef OMITBAD

void CWE23_Relative_Path_Traversal__char_fromConsole_fopen_32_bad()
{
    char * data;
    char * *data_ptr1 = &data;
    char * *data_ptr2 = &data;
    char data_buf[FILENAME_MAX] = BASEPATH;
    data = data_buf;
    {
        char * data = *data_ptr1;
        {
            /* Read input from the console */
            size_t data_len = strlen(data);
            /* if there is room in data, read into it from the console */
            if(FILENAME_MAX-data_len > 1)
            {
                fgets(data+data_len, (int)(FILENAME_MAX-data_len), stdin);
                /* The next 3 lines remove the carriage return from the string that is
                 * inserted by fgets() */
                data_len = strlen(data);
                if (data_len > 0)
                {
                    data[data_len-1] = '\0';
                }
            }
        }
        *data_ptr1 = data;
    }
    {
        char * data = *data_ptr2;
        {
            FILE *file = NULL;
            /* POTENTIAL FLAW: Possibly opening a file without validating the file name or path */
            file = FOPEN(data, "wb+");
            if (file != NULL) fclose(file);
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
    char data_buf[FILENAME_MAX] = BASEPATH;
    data = data_buf;
    {
        char * data = *data_ptr1;
        /* FIX: File name does not contain a period or slash */
        strcat(data, "file.txt");
        *data_ptr1 = data;
    }
    {
        char * data = *data_ptr2;
        {
            FILE *file = NULL;
            /* POTENTIAL FLAW: Possibly opening a file without validating the file name or path */
            file = FOPEN(data, "wb+");
            if (file != NULL) fclose(file);
        }
    }
}

void CWE23_Relative_Path_Traversal__char_fromConsole_fopen_32_good()
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
    CWE23_Relative_Path_Traversal__char_fromConsole_fopen_32_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE23_Relative_Path_Traversal__char_fromConsole_fopen_32_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
