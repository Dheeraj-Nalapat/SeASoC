/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE187_Partial_Comparison__wchar_t_fromFile_str_65a.c
Label Definition File: CWE187_Partial_Comparison.label.xml
Template File: sources-sinks-65a.tmpl.c
*/
/*
 * @description
 * CWE: 187 Partial Comparison
 * BadSource: fromFile Read input from a file
 * GoodSource: Provide a matching password
 * Sinks: str
 *    GoodSink: Compare the 2 passwords correctly
 *    BadSink : use wcsstr() to do password match, which is a partial comparison
 * Flow Variant: 65 Data/control flow: data passed as an argument from one function to a function in a different source file called via a function pointer
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#define PASSWORD L"Password1234"
/* PASSWORD_SZ must equal the length of PASSWORD */
#define PASSWORD_SZ wcslen(PASSWORD)

#ifdef _WIN32
# define FOPEN _wfopen
#else
/* fopen is used on unix-based OSs */
# define FOPEN fopen
#endif

#ifndef OMITBAD

/* bad function declaration */
void CWE187_Partial_Comparison__wchar_t_fromFile_str_65b_bad_sink(wchar_t * data);

void CWE187_Partial_Comparison__wchar_t_fromFile_str_65_bad()
{
    wchar_t * data;
    /* define a function pointer */
    void (*func_ptr) (wchar_t *) = CWE187_Partial_Comparison__wchar_t_fromFile_str_65b_bad_sink;
    wchar_t data_buf[100] = L"";
    data = data_buf;
    {
        /* Read input from a file */
        size_t data_len = wcslen(data);
        FILE * pFile;
        /* if there is room in data, attempt to read the input from a file */
        if(100-data_len > 1)
        {
            pFile = FOPEN(L"C:\\temp\\file.txt", L"r");
            if (pFile != NULL)
            {
                fgetws(data+data_len, (int)(100-data_len), pFile);
                fclose(pFile);
            }
        }
    }
    /* use the function pointer */
    func_ptr(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE187_Partial_Comparison__wchar_t_fromFile_str_65b_goodG2B_sink(wchar_t * data);

static void goodG2B()
{
    wchar_t * data;
    void (*func_ptr) (wchar_t *) = CWE187_Partial_Comparison__wchar_t_fromFile_str_65b_goodG2B_sink;
    wchar_t data_buf[100] = L"";
    data = data_buf;
    /* FIX: Use the matching string */
    data = PASSWORD;
    func_ptr(data);
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE187_Partial_Comparison__wchar_t_fromFile_str_65b_goodB2G_sink(wchar_t * data);

static void goodB2G()
{
    wchar_t * data;
    void (*func_ptr) (wchar_t *) = CWE187_Partial_Comparison__wchar_t_fromFile_str_65b_goodB2G_sink;
    wchar_t data_buf[100] = L"";
    data = data_buf;
    {
        /* Read input from a file */
        size_t data_len = wcslen(data);
        FILE * pFile;
        /* if there is room in data, attempt to read the input from a file */
        if(100-data_len > 1)
        {
            pFile = FOPEN(L"C:\\temp\\file.txt", L"r");
            if (pFile != NULL)
            {
                fgetws(data+data_len, (int)(100-data_len), pFile);
                fclose(pFile);
            }
        }
    }
    func_ptr(data);
}

void CWE187_Partial_Comparison__wchar_t_fromFile_str_65_good()
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
    CWE187_Partial_Comparison__wchar_t_fromFile_str_65_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE187_Partial_Comparison__wchar_t_fromFile_str_65_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
