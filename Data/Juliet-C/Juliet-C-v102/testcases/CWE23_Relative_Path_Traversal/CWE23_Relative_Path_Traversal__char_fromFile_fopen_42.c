/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE23_Relative_Path_Traversal__char_fromFile_fopen_42.c
Label Definition File: CWE23_Relative_Path_Traversal.label.xml
Template File: sources-sink-42.tmpl.c
*/
/*
 * @description
 * CWE: 23 Relative Path Traversal
 * BadSource: fromFile Read input from a file
 * GoodSource: File name without a period or slash
 * Sink: fopen
 *    BadSink :
 * Flow Variant: 42 Data flow: data returned from one function to another in the same source file
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

#ifdef _WIN32
# define FOPEN fopen
#else
/* fopen is used on unix-based OSs */
# define FOPEN fopen
#endif

#ifndef OMITBAD

static char * bad_source(char * data)
{
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
    return data;
}

void CWE23_Relative_Path_Traversal__char_fromFile_fopen_42_bad()
{
    char * data;
    char data_buf[FILENAME_MAX] = BASEPATH;
    data = data_buf;
    data = bad_source(data);
    {
        FILE *file = NULL;
        /* POTENTIAL FLAW: Possibly opening a file without validating the file name or path */
        file = FOPEN(data, "wb+");
        if (file != NULL) fclose(file);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

static char * goodG2B_source(char * data)
{
    /* FIX: File name does not contain a period or slash */
    strcat(data, "file.txt");
    return data;
}

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    char * data;
    char data_buf[FILENAME_MAX] = BASEPATH;
    data = data_buf;
    data = goodG2B_source(data);
    {
        FILE *file = NULL;
        /* POTENTIAL FLAW: Possibly opening a file without validating the file name or path */
        file = FOPEN(data, "wb+");
        if (file != NULL) fclose(file);
    }
}

void CWE23_Relative_Path_Traversal__char_fromFile_fopen_42_good()
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
    CWE23_Relative_Path_Traversal__char_fromFile_fopen_42_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE23_Relative_Path_Traversal__char_fromFile_fopen_42_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
