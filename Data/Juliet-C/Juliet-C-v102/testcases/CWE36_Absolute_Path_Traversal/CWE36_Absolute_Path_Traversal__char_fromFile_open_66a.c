/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE36_Absolute_Path_Traversal__char_fromFile_open_66a.c
Label Definition File: CWE36_Absolute_Path_Traversal.label.xml
Template File: sources-sink-66a.tmpl.c
*/
/*
 * @description
 * CWE: 36 Absolute Path Traversal
 * BadSource: fromFile Read input from a file
 * GoodSource: Full path and file name
 * Sinks: open
 *    BadSink :
 * Flow Variant: 66 Data flow: data passed in an array from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# define FOPEN fopen
#else
/* fopen is used on unix-based OSs */
# define FOPEN fopen
#endif

#ifdef _WIN32
# define OPEN _open
# define CLOSE _close
#else
# define OPEN open
# define CLOSE close
#endif

#ifndef OMITBAD

/* bad function declaration */
void CWE36_Absolute_Path_Traversal__char_fromFile_open_66b_bad_sink(char * data_array[]);

void CWE36_Absolute_Path_Traversal__char_fromFile_open_66_bad()
{
    char * data;
    char * data_array[5];
    char data_buf[FILENAME_MAX] = "";
    data = data_buf;
    {
        /* Read input from a file */
        size_t data_len = strlen(data);
        FILE * pFile;
        /* if there is room in data, attempt to read the input from a file */
        if(FILENAME_MAX-data_len > 1)
        {
            pFile = FOPEN("C:\\temp\\file.txt", "r");
            if (pFile != NULL)
            {
                fgets(data+data_len, (int)(FILENAME_MAX-data_len), pFile);
                fclose(pFile);
            }
        }
    }
    /* put data in array */
    data_array[2] = data;
    CWE36_Absolute_Path_Traversal__char_fromFile_open_66b_bad_sink(data_array);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE36_Absolute_Path_Traversal__char_fromFile_open_66b_goodG2B_sink(char * data_array[]);

static void goodG2B()
{
    char * data;
    char * data_array[5];
    char data_buf[FILENAME_MAX] = "";
    data = data_buf;
#ifdef _WIN32
    /* FIX: Full path and file name */
    strcpy(data, "c:\\temp\\file.txt");
#else
    /* FIX: Full path and file name */
    strcpy(data, "/tmp/file.txt");
#endif
    data_array[2] = data;
    CWE36_Absolute_Path_Traversal__char_fromFile_open_66b_goodG2B_sink(data_array);
}

void CWE36_Absolute_Path_Traversal__char_fromFile_open_66_good()
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
    CWE36_Absolute_Path_Traversal__char_fromFile_open_66_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE36_Absolute_Path_Traversal__char_fromFile_open_66_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
