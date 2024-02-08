/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE127_Buffer_Underread__char_declare_cpy_63a.c
Label Definition File: CWE127_Buffer_Underread.stack.label.xml
Template File: sources-sink-63a.tmpl.c
*/
/*
 * @description
 * CWE: 127 Buffer Under-read
 * BadSource:  Set data pointer to before the allocated memory buffer
 * GoodSource: Set data pointer to the allocated memory buffer
 * Sinks: cpy
 *    BadSink : Copy data to string using strcpy
 * Flow Variant: 63 Data flow: pointer to data passed from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

/* bad function declaration */
void CWE127_Buffer_Underread__char_declare_cpy_63b_bad_sink(char * * data_ptr);

void CWE127_Buffer_Underread__char_declare_cpy_63_bad()
{
    char * data;
    char data_buf[100];
    memset(data_buf, 'A', 100-1);
    data_buf[100-1] = '\0';
    /* FLAW: Set data pointer to before the allocated memory buffer */
    data = data_buf - 8;
    CWE127_Buffer_Underread__char_declare_cpy_63b_bad_sink(&data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE127_Buffer_Underread__char_declare_cpy_63b_goodG2B_sink(char * * data);

static void goodG2B()
{
    char * data;
    char data_buf[100];
    memset(data_buf, 'A', 100-1);
    data_buf[100-1] = '\0';
    /* FIX: Set data pointer to the allocated memory buffer */
    data = data_buf;
    CWE127_Buffer_Underread__char_declare_cpy_63b_goodG2B_sink(&data);
}

void CWE127_Buffer_Underread__char_declare_cpy_63_good()
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
    CWE127_Buffer_Underread__char_declare_cpy_63_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE127_Buffer_Underread__char_declare_cpy_63_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
