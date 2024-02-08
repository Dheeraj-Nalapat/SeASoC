/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE124_Buffer_Underwrite__char_alloca_memcpy_65a.c
Label Definition File: CWE124_Buffer_Underwrite.stack.label.xml
Template File: sources-sink-65a.tmpl.c
*/
/*
 * @description
 * CWE: 124 Buffer Underwrite
 * BadSource:  Set data pointer to before the allocated memory buffer
 * GoodSource: Set data pointer to the allocated memory buffer
 * Sinks: memcpy
 *    BadSink : Copy string to data using memcpy
 * Flow Variant: 65 Data/control flow: data passed as an argument from one function to a function in a different source file called via a function pointer
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

/* bad function declaration */
void CWE124_Buffer_Underwrite__char_alloca_memcpy_65b_bad_sink(char * data);

void CWE124_Buffer_Underwrite__char_alloca_memcpy_65_bad()
{
    char * data;
    /* define a function pointer */
    void (*func_ptr) (char *) = CWE124_Buffer_Underwrite__char_alloca_memcpy_65b_bad_sink;
    char * data_buf = (char *)ALLOCA(100*sizeof(char));
    memset(data_buf, 'A', 100-1);
    data_buf[100-1] = '\0';
    /* FLAW: Set data pointer to before the allocated memory buffer */
    data = data_buf - 8;
    /* use the function pointer */
    func_ptr(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE124_Buffer_Underwrite__char_alloca_memcpy_65b_goodG2B_sink(char * data);

static void goodG2B()
{
    char * data;
    void (*func_ptr) (char *) = CWE124_Buffer_Underwrite__char_alloca_memcpy_65b_goodG2B_sink;
    char * data_buf = (char *)ALLOCA(100*sizeof(char));
    memset(data_buf, 'A', 100-1);
    data_buf[100-1] = '\0';
    /* FIX: Set data pointer to the allocated memory buffer */
    data = data_buf;
    func_ptr(data);
}

void CWE124_Buffer_Underwrite__char_alloca_memcpy_65_good()
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
    CWE124_Buffer_Underwrite__char_alloca_memcpy_65_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE124_Buffer_Underwrite__char_alloca_memcpy_65_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
