/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE23_Relative_Path_Traversal__wchar_t_fromFile_open_44.c
Label Definition File: CWE23_Relative_Path_Traversal.label.xml
Template File: sources-sink-44.tmpl.c
*/
/*
 * @description
 * CWE: 23 Relative Path Traversal
 * BadSource: fromFile Read input from a file
 * GoodSource: File name without a period or slash
 * Sinks: open
 *    BadSink :
 * Flow Variant: 44 Data/control flow: data passed as an argument from one function to a function in the same source file called via a function pointer
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
#define BASEPATH L"c:\\temp\\"
#else
#define BASEPATH L"/tmp/"
#endif

#ifdef _WIN32
# define FOPEN _wfopen
#else
/* fopen is used on unix-based OSs */
# define FOPEN fopen
#endif

#ifdef _WIN32
# define OPEN _wopen
# define CLOSE _close
#else
# define OPEN wopen
# define CLOSE close
#endif

#ifndef OMITBAD

static void bad_sink(wchar_t * data)
{
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

void CWE23_Relative_Path_Traversal__wchar_t_fromFile_open_44_bad()
{
    wchar_t * data;
    /* define a function pointer */
    void (*func_ptr) (wchar_t *) = bad_sink;
    wchar_t data_buf[FILENAME_MAX] = BASEPATH;
    data = data_buf;
    {
        /* Read input from a file */
        size_t data_len = wcslen(data);
        FILE * pFile;
        /* if there is room in data, attempt to read the input from a file */
        if(FILENAME_MAX-data_len > 1)
        {
            pFile = FOPEN(L"C:\\temp\\file.txt", L"r");
            if (pFile != NULL)
            {
                fgetws(data+data_len, (int)(FILENAME_MAX-data_len), pFile);
                fclose(pFile);
            }
        }
    }
    /* use the function pointer */
    func_ptr(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B_sink(wchar_t * data)
{
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

static void goodG2B()
{
    wchar_t * data;
    void (*func_ptr) (wchar_t *) = goodG2B_sink;
    wchar_t data_buf[FILENAME_MAX] = BASEPATH;
    data = data_buf;
    /* FIX: File name does not contain a period or slash */
    wcscat(data, L"file.txt");
    func_ptr(data);
}

void CWE23_Relative_Path_Traversal__wchar_t_fromFile_open_44_good()
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
    CWE23_Relative_Path_Traversal__wchar_t_fromFile_open_44_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE23_Relative_Path_Traversal__wchar_t_fromFile_open_44_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
