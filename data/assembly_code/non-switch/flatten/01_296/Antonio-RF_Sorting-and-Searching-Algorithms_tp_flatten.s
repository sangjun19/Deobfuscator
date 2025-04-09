	.file	"Antonio-RF_Sorting-and-Searching-Algorithms_tp_flatten.c"
	.text
	.globl	_TIG_IZ_0hIZ_argc
	.bss
	.align 4
	.type	_TIG_IZ_0hIZ_argc, @object
	.size	_TIG_IZ_0hIZ_argc, 4
_TIG_IZ_0hIZ_argc:
	.zero	4
	.globl	_TIG_IZ_0hIZ_envp
	.align 8
	.type	_TIG_IZ_0hIZ_envp, @object
	.size	_TIG_IZ_0hIZ_envp, 8
_TIG_IZ_0hIZ_envp:
	.zero	8
	.globl	count_trocas
	.align 8
	.type	count_trocas, @object
	.size	count_trocas, 8
count_trocas:
	.zero	8
	.globl	count_comparacoes
	.align 8
	.type	count_comparacoes, @object
	.size	count_comparacoes, 8
count_comparacoes:
	.zero	8
	.globl	_TIG_IZ_0hIZ_argv
	.align 8
	.type	_TIG_IZ_0hIZ_argv, @object
	.size	_TIG_IZ_0hIZ_argv, 8
_TIG_IZ_0hIZ_argv:
	.zero	8
	.globl	vector_das_comparacoes
	.align 32
	.type	vector_das_comparacoes, @object
	.size	vector_das_comparacoes, 4000
vector_das_comparacoes:
	.zero	4000
	.text
	.globl	selection_sort
	.type	selection_sort, @function
selection_sort:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movq	$11, -8(%rbp)
.L22:
	cmpq	$14, -8(%rbp)
	ja	.L24
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L24-.L4
	.long	.L8-.L4
	.long	.L24-.L4
	.long	.L24-.L4
	.long	.L24-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L24-.L4
	.long	.L3-.L4
	.text
.L9:
	movq	count_comparacoes(%rip), %rax
	addq	$1, %rax
	movq	%rax, count_comparacoes(%rip)
	movq	$10, -8(%rbp)
	jmp	.L14
.L3:
	movl	-20(%rbp), %eax
	movl	%eax, -12(%rbp)
	movl	-20(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L14
.L5:
	movq	count_comparacoes(%rip), %rax
	jmp	.L23
.L12:
	movl	-16(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jge	.L16
	movq	$4, -8(%rbp)
	jmp	.L14
.L16:
	movq	$6, -8(%rbp)
	jmp	.L14
.L10:
	addl	$1, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L14
.L6:
	movl	$0, -20(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L14
.L8:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-40(%rbp), %rax
	addq	%rcx, %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	troca
	movq	count_trocas(%rip), %rax
	addq	$1, %rax
	movq	%rax, count_trocas(%rip)
	addl	$1, -20(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L14
.L7:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-40(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jge	.L18
	movq	$0, -8(%rbp)
	jmp	.L14
.L18:
	movq	$3, -8(%rbp)
	jmp	.L14
.L13:
	movl	-16(%rbp), %eax
	movl	%eax, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L14
.L11:
	movl	-44(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -20(%rbp)
	jge	.L20
	movq	$14, -8(%rbp)
	jmp	.L14
.L20:
	movq	$12, -8(%rbp)
	jmp	.L14
.L24:
	nop
.L14:
	jmp	.L22
.L23:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	selection_sort, .-selection_sort
	.section	.rodata
	.align 8
.LC0:
	.string	"M\303\251dia da Pesquisa Bin\303\241ria: %lld\n"
	.align 8
.LC1:
	.string	"Desvio Padr\303\243o da Pesquisa Bin\303\241ria: %lld\n"
	.text
	.globl	mil_pesquisa_binaria
	.type	mil_pesquisa_binaria, @function
mil_pesquisa_binaria:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$64, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$2, -8240(%rbp)
.L38:
	cmpq	$9, -8240(%rbp)
	ja	.L41
	movq	-8240(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L28(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L28(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L28:
	.long	.L33-.L28
	.long	.L41-.L28
	.long	.L32-.L28
	.long	.L31-.L28
	.long	.L41-.L28
	.long	.L30-.L28
	.long	.L41-.L28
	.long	.L41-.L28
	.long	.L42-.L28
	.long	.L27-.L28
	.text
.L31:
	leaq	vector_das_comparacoes(%rip), %rax
	movq	%rax, %rdi
	call	media
	movl	%eax, -8248(%rbp)
	movl	-8248(%rbp), %eax
	cltq
	movq	%rax, -8224(%rbp)
	leaq	vector_das_comparacoes(%rip), %rax
	movq	%rax, %rdi
	call	desvio_padrao
	movl	%eax, -8244(%rbp)
	movl	-8244(%rbp), %eax
	cltq
	movq	%rax, -8216(%rbp)
	movq	-8224(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-8216(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -8240(%rbp)
	jmp	.L35
.L27:
	leaq	-4112(%rbp), %rdx
	leaq	-8208(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	cria_vector
	movl	$2048, %esi
	movl	$0, %edi
	call	aleat
	movq	%rax, -8232(%rbp)
	movq	-8232(%rbp), %rax
	movl	%eax, -8252(%rbp)
	leaq	-8208(%rbp), %rax
	movl	$2, %ecx
	movl	$1023, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	quick_sort
	movq	$0, count_comparacoes(%rip)
	movl	-8252(%rbp), %edx
	leaq	-8208(%rbp), %rax
	movl	%edx, %ecx
	movl	$1024, %edx
	movl	$2, %esi
	movq	%rax, %rdi
	call	pesquisa_binaria
	movq	count_comparacoes(%rip), %rax
	movl	%eax, %ecx
	movl	-8256(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	vector_das_comparacoes(%rip), %rax
	movl	%ecx, (%rdx,%rax)
	movq	$0, count_comparacoes(%rip)
	addl	$1, -8256(%rbp)
	movq	$5, -8240(%rbp)
	jmp	.L35
.L30:
	cmpl	$999, -8256(%rbp)
	jg	.L36
	movq	$9, -8240(%rbp)
	jmp	.L35
.L36:
	movq	$3, -8240(%rbp)
	jmp	.L35
.L33:
	movq	$0, count_comparacoes(%rip)
	movl	$0, -8256(%rbp)
	movq	$5, -8240(%rbp)
	jmp	.L35
.L32:
	movq	$0, -8240(%rbp)
	jmp	.L35
.L41:
	nop
.L35:
	jmp	.L38
.L42:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L40
	call	__stack_chk_fail@PLT
.L40:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	mil_pesquisa_binaria, .-mil_pesquisa_binaria
	.section	.rodata
.LC2:
	.string	"%d "
	.text
	.globl	imprime_parte
	.type	imprime_parte, @function
imprime_parte:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$1, -8(%rbp)
.L55:
	cmpq	$6, -8(%rbp)
	ja	.L56
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L46(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L46(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L46:
	.long	.L50-.L46
	.long	.L49-.L46
	.long	.L56-.L46
	.long	.L56-.L46
	.long	.L57-.L46
	.long	.L47-.L46
	.long	.L45-.L46
	.text
.L49:
	movl	$0, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L52
.L45:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L52
.L47:
	cmpl	$99, -12(%rbp)
	jg	.L53
	movq	$6, -8(%rbp)
	jmp	.L52
.L53:
	movq	$0, -8(%rbp)
	jmp	.L52
.L50:
	movl	$10, %edi
	call	putchar@PLT
	movq	$4, -8(%rbp)
	jmp	.L52
.L56:
	nop
.L52:
	jmp	.L55
.L57:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	imprime_parte, .-imprime_parte
	.section	.rodata
	.align 8
.LC3:
	.string	"Entrada n\303\243o compreendida. Caso deseja fazer a pesquisa sequencial, aperte 6 novamente!"
	.text
	.globl	pesquisa_sequencial
	.type	pesquisa_sequencial, @function
pesquisa_sequencial:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movl	%edx, -32(%rbp)
	movl	%ecx, -36(%rbp)
	movq	$5, -8(%rbp)
.L81:
	cmpq	$12, -8(%rbp)
	ja	.L82
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L61(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L61(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L61:
	.long	.L70-.L61
	.long	.L82-.L61
	.long	.L69-.L61
	.long	.L82-.L61
	.long	.L68-.L61
	.long	.L67-.L61
	.long	.L66-.L61
	.long	.L65-.L61
	.long	.L82-.L61
	.long	.L64-.L61
	.long	.L63-.L61
	.long	.L62-.L61
	.long	.L60-.L61
	.text
.L68:
	movl	$-1, %eax
	jmp	.L71
.L60:
	movl	-12(%rbp), %eax
	jmp	.L71
.L62:
	movl	-12(%rbp), %eax
	cmpl	-32(%rbp), %eax
	jge	.L72
	movq	$10, -8(%rbp)
	jmp	.L74
.L72:
	movq	$0, -8(%rbp)
	jmp	.L74
.L64:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -8(%rbp)
	jmp	.L74
.L66:
	movq	count_comparacoes(%rip), %rax
	addq	$1, %rax
	movq	%rax, count_comparacoes(%rip)
	addl	$1, -12(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L74
.L67:
	cmpl	$1, -28(%rbp)
	jne	.L75
	movq	$7, -8(%rbp)
	jmp	.L74
.L75:
	movq	$2, -8(%rbp)
	jmp	.L74
.L63:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -36(%rbp)
	jne	.L77
	movq	$12, -8(%rbp)
	jmp	.L74
.L77:
	movq	$6, -8(%rbp)
	jmp	.L74
.L70:
	movl	$-1, %eax
	jmp	.L71
.L65:
	movl	$0, -12(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L74
.L69:
	cmpl	$2, -28(%rbp)
	jne	.L79
	movq	$7, -8(%rbp)
	jmp	.L74
.L79:
	movq	$9, -8(%rbp)
	jmp	.L74
.L82:
	nop
.L74:
	jmp	.L81
.L71:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	pesquisa_sequencial, .-pesquisa_sequencial
	.section	.rodata
	.align 8
.LC4:
	.string	"M\303\251dia do Shell Sort por espa\303\247amento padr\303\243o: %lld\n"
	.align 8
.LC5:
	.string	"Desvio padrao do Shell Sort por espa\303\247amento padr\303\243o: %lld\n"
	.text
	.globl	mil_shell_sort_padrao
	.type	mil_shell_sort_padrao, @function
mil_shell_sort_padrao:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$64, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$6, -8232(%rbp)
.L96:
	cmpq	$9, -8232(%rbp)
	ja	.L99
	movq	-8232(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L86(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L86(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L86:
	.long	.L99-.L86
	.long	.L99-.L86
	.long	.L91-.L86
	.long	.L99-.L86
	.long	.L99-.L86
	.long	.L90-.L86
	.long	.L89-.L86
	.long	.L100-.L86
	.long	.L87-.L86
	.long	.L85-.L86
	.text
.L87:
	cmpl	$999, -8244(%rbp)
	jg	.L92
	movq	$2, -8232(%rbp)
	jmp	.L94
.L92:
	movq	$9, -8232(%rbp)
	jmp	.L94
.L85:
	leaq	vector_das_comparacoes(%rip), %rax
	movq	%rax, %rdi
	call	media
	movl	%eax, -8240(%rbp)
	movl	-8240(%rbp), %eax
	cltq
	movq	%rax, -8224(%rbp)
	leaq	vector_das_comparacoes(%rip), %rax
	movq	%rax, %rdi
	call	desvio_padrao
	movl	%eax, -8236(%rbp)
	movl	-8236(%rbp), %eax
	cltq
	movq	%rax, -8216(%rbp)
	movq	-8224(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-8216(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -8232(%rbp)
	jmp	.L94
.L89:
	movq	$5, -8232(%rbp)
	jmp	.L94
.L90:
	movq	$0, count_comparacoes(%rip)
	movl	$0, -8244(%rbp)
	movq	$8, -8232(%rbp)
	jmp	.L94
.L91:
	leaq	-4112(%rbp), %rdx
	leaq	-8208(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	cria_vector
	leaq	-8208(%rbp), %rax
	movl	$2, %edx
	movl	$1024, %esi
	movq	%rax, %rdi
	call	shell_sort
	movq	count_comparacoes(%rip), %rax
	movl	%eax, %ecx
	movl	-8244(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	vector_das_comparacoes(%rip), %rax
	movl	%ecx, (%rdx,%rax)
	movq	$0, count_comparacoes(%rip)
	addl	$1, -8244(%rbp)
	movq	$8, -8232(%rbp)
	jmp	.L94
.L99:
	nop
.L94:
	jmp	.L96
.L100:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L98
	call	__stack_chk_fail@PLT
.L98:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	mil_shell_sort_padrao, .-mil_shell_sort_padrao
	.section	.rodata
	.align 8
.LC6:
	.string	"M\303\251dia do Selection Sort: %lld\n"
	.align 8
.LC7:
	.string	"Desvio padr\303\243o do Selection Sort: %lld\n"
	.text
	.globl	mil_selection_sort
	.type	mil_selection_sort, @function
mil_selection_sort:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$64, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, -8232(%rbp)
.L114:
	cmpq	$8, -8232(%rbp)
	ja	.L117
	movq	-8232(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L104(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L104(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L104:
	.long	.L109-.L104
	.long	.L118-.L104
	.long	.L117-.L104
	.long	.L107-.L104
	.long	.L106-.L104
	.long	.L117-.L104
	.long	.L105-.L104
	.long	.L117-.L104
	.long	.L103-.L104
	.text
.L106:
	cmpl	$999, -8244(%rbp)
	jg	.L110
	movq	$6, -8232(%rbp)
	jmp	.L112
.L110:
	movq	$3, -8232(%rbp)
	jmp	.L112
.L103:
	movq	$0, count_comparacoes(%rip)
	movl	$0, -8244(%rbp)
	movq	$4, -8232(%rbp)
	jmp	.L112
.L107:
	leaq	vector_das_comparacoes(%rip), %rax
	movq	%rax, %rdi
	call	media
	movl	%eax, -8240(%rbp)
	movl	-8240(%rbp), %eax
	cltq
	movq	%rax, -8224(%rbp)
	leaq	vector_das_comparacoes(%rip), %rax
	movq	%rax, %rdi
	call	desvio_padrao
	movl	%eax, -8236(%rbp)
	movl	-8236(%rbp), %eax
	cltq
	movq	%rax, -8216(%rbp)
	movq	-8224(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-8216(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8232(%rbp)
	jmp	.L112
.L105:
	leaq	-4112(%rbp), %rdx
	leaq	-8208(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	cria_vector
	leaq	-8208(%rbp), %rax
	movl	$1024, %esi
	movq	%rax, %rdi
	call	selection_sort
	movl	-8244(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	leaq	vector_das_comparacoes(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	movq	$0, count_comparacoes(%rip)
	addl	$1, -8244(%rbp)
	movq	$4, -8232(%rbp)
	jmp	.L112
.L109:
	movq	$8, -8232(%rbp)
	jmp	.L112
.L117:
	nop
.L112:
	jmp	.L114
.L118:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L116
	call	__stack_chk_fail@PLT
.L116:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	mil_selection_sort, .-mil_selection_sort
	.globl	shell_sort
	.type	shell_sort, @function
shell_sort:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -56(%rbp)
	movl	%esi, -60(%rbp)
	movl	%edx, -64(%rbp)
	movq	$13, -8(%rbp)
.L183:
	cmpq	$51, -8(%rbp)
	ja	.L184
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L122(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L122(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L122:
	.long	.L154-.L122
	.long	.L153-.L122
	.long	.L184-.L122
	.long	.L152-.L122
	.long	.L151-.L122
	.long	.L150-.L122
	.long	.L149-.L122
	.long	.L148-.L122
	.long	.L147-.L122
	.long	.L184-.L122
	.long	.L146-.L122
	.long	.L145-.L122
	.long	.L144-.L122
	.long	.L143-.L122
	.long	.L142-.L122
	.long	.L141-.L122
	.long	.L140-.L122
	.long	.L139-.L122
	.long	.L138-.L122
	.long	.L137-.L122
	.long	.L184-.L122
	.long	.L136-.L122
	.long	.L135-.L122
	.long	.L184-.L122
	.long	.L184-.L122
	.long	.L184-.L122
	.long	.L184-.L122
	.long	.L134-.L122
	.long	.L133-.L122
	.long	.L184-.L122
	.long	.L132-.L122
	.long	.L184-.L122
	.long	.L184-.L122
	.long	.L131-.L122
	.long	.L184-.L122
	.long	.L184-.L122
	.long	.L184-.L122
	.long	.L184-.L122
	.long	.L184-.L122
	.long	.L184-.L122
	.long	.L130-.L122
	.long	.L184-.L122
	.long	.L184-.L122
	.long	.L129-.L122
	.long	.L128-.L122
	.long	.L127-.L122
	.long	.L126-.L122
	.long	.L125-.L122
	.long	.L124-.L122
	.long	.L184-.L122
	.long	.L123-.L122
	.long	.L185-.L122
	.text
.L138:
	movq	count_trocas(%rip), %rax
	addq	$1, %rax
	movq	%rax, count_trocas(%rip)
	movq	$17, -8(%rbp)
	jmp	.L155
.L123:
	movl	-12(%rbp), %eax
	cmpl	-36(%rbp), %eax
	jl	.L156
	movq	$6, -8(%rbp)
	jmp	.L155
.L156:
	movq	$43, -8(%rbp)
	jmp	.L155
.L151:
	movl	-36(%rbp), %eax
	movl	%eax, -20(%rbp)
	movq	$33, -8(%rbp)
	jmp	.L155
.L132:
	movl	-24(%rbp), %eax
	cmpl	-36(%rbp), %eax
	jl	.L158
	movq	$48, -8(%rbp)
	jmp	.L155
.L158:
	movq	$16, -8(%rbp)
	jmp	.L155
.L142:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -28(%rbp)
	movl	-32(%rbp), %eax
	movl	%eax, -24(%rbp)
	movq	$30, -8(%rbp)
	jmp	.L155
.L141:
	movl	-12(%rbp), %eax
	subl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -16(%rbp)
	jl	.L160
	movq	$43, -8(%rbp)
	jmp	.L155
.L160:
	movq	$0, -8(%rbp)
	jmp	.L155
.L144:
	movl	$1, -36(%rbp)
	movq	$40, -8(%rbp)
	jmp	.L155
.L147:
	movl	-24(%rbp), %eax
	cmpl	-32(%rbp), %eax
	je	.L162
	movq	$18, -8(%rbp)
	jmp	.L155
.L162:
	movq	$17, -8(%rbp)
	jmp	.L155
.L127:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$1431655766, %rdx, %rdx
	shrq	$32, %rdx
	sarl	$31, %eax
	movl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, -36(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L155
.L153:
	cmpl	$0, -36(%rbp)
	jle	.L164
	movq	$3, -8(%rbp)
	jmp	.L155
.L164:
	movq	$51, -8(%rbp)
	jmp	.L155
.L152:
	movl	-36(%rbp), %eax
	movl	%eax, -32(%rbp)
	movq	$21, -8(%rbp)
	jmp	.L155
.L140:
	movl	-24(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movl	-28(%rbp), %eax
	movl	%eax, (%rdx)
	movq	$8, -8(%rbp)
	jmp	.L155
.L136:
	movl	-32(%rbp), %eax
	cmpl	-60(%rbp), %eax
	jge	.L166
	movq	$14, -8(%rbp)
	jmp	.L155
.L166:
	movq	$19, -8(%rbp)
	jmp	.L155
.L145:
	movl	-36(%rbp), %edx
	movl	%edx, %eax
	addl	%eax, %eax
	addl	%edx, %eax
	addl	$1, %eax
	movl	%eax, -36(%rbp)
	movq	$40, -8(%rbp)
	jmp	.L155
.L143:
	cmpl	$1, -64(%rbp)
	jne	.L168
	movq	$27, -8(%rbp)
	jmp	.L155
.L168:
	movq	$46, -8(%rbp)
	jmp	.L155
.L137:
	movl	-36(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	movl	%eax, -36(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L155
.L139:
	addl	$1, -32(%rbp)
	movq	$21, -8(%rbp)
	jmp	.L155
.L130:
	movl	-60(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$1431655766, %rdx, %rdx
	shrq	$32, %rdx
	sarl	$31, %eax
	movl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	cmpl	%eax, -36(%rbp)
	jge	.L171
	movq	$11, -8(%rbp)
	jmp	.L155
.L171:
	movq	$7, -8(%rbp)
	jmp	.L155
.L149:
	movq	count_comparacoes(%rip), %rax
	addq	$1, %rax
	movq	%rax, count_comparacoes(%rip)
	movq	$15, -8(%rbp)
	jmp	.L155
.L134:
	movl	-60(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	movl	%eax, -36(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L155
.L124:
	movq	count_comparacoes(%rip), %rax
	addq	$1, %rax
	movq	%rax, count_comparacoes(%rip)
	movq	$5, -8(%rbp)
	jmp	.L155
.L135:
	movl	-12(%rbp), %eax
	cmpl	-20(%rbp), %eax
	je	.L173
	movq	$28, -8(%rbp)
	jmp	.L155
.L173:
	movq	$44, -8(%rbp)
	jmp	.L155
.L133:
	movq	count_trocas(%rip), %rax
	addq	$1, %rax
	movq	%rax, count_trocas(%rip)
	movq	$44, -8(%rbp)
	jmp	.L155
.L125:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -16(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, -12(%rbp)
	movq	$50, -8(%rbp)
	jmp	.L155
.L128:
	addl	$1, -20(%rbp)
	movq	$33, -8(%rbp)
	jmp	.L155
.L150:
	movl	-24(%rbp), %eax
	subl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -28(%rbp)
	jl	.L175
	movq	$16, -8(%rbp)
	jmp	.L155
.L175:
	movq	$10, -8(%rbp)
	jmp	.L155
.L131:
	movl	-20(%rbp), %eax
	cmpl	-60(%rbp), %eax
	jge	.L177
	movq	$47, -8(%rbp)
	jmp	.L155
.L177:
	movq	$45, -8(%rbp)
	jmp	.L155
.L146:
	movl	-24(%rbp), %eax
	subl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	-24(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	movq	count_trocas(%rip), %rax
	addq	$1, %rax
	movq	%rax, count_trocas(%rip)
	movl	-36(%rbp), %eax
	subl	%eax, -24(%rbp)
	movq	$30, -8(%rbp)
	jmp	.L155
.L154:
	movl	-12(%rbp), %eax
	subl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	-12(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	movq	count_trocas(%rip), %rax
	addq	$1, %rax
	movq	%rax, count_trocas(%rip)
	movl	-36(%rbp), %eax
	subl	%eax, -12(%rbp)
	movq	$50, -8(%rbp)
	jmp	.L155
.L126:
	cmpl	$2, -64(%rbp)
	jne	.L179
	movq	$12, -8(%rbp)
	jmp	.L155
.L179:
	movq	$51, -8(%rbp)
	jmp	.L155
.L148:
	cmpl	$0, -36(%rbp)
	jle	.L181
	movq	$4, -8(%rbp)
	jmp	.L155
.L181:
	movq	$51, -8(%rbp)
	jmp	.L155
.L129:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movl	-16(%rbp), %eax
	movl	%eax, (%rdx)
	movq	$22, -8(%rbp)
	jmp	.L155
.L184:
	nop
.L155:
	jmp	.L183
.L185:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	shell_sort, .-shell_sort
	.section	.rodata
	.align 8
.LC8:
	.string	"M\303\251dia do Shell Sort por espa\303\247amento de Knuth: %lld\n"
	.align 8
.LC9:
	.string	"Desvio padrao do Shell Sort por espa\303\247amento de Knuth: %lld\n"
	.text
	.globl	mil_shell_sort_knuth
	.type	mil_shell_sort_knuth, @function
mil_shell_sort_knuth:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$64, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$9, -8232(%rbp)
.L199:
	cmpq	$9, -8232(%rbp)
	ja	.L202
	movq	-8232(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L189(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L189(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L189:
	.long	.L194-.L189
	.long	.L202-.L189
	.long	.L193-.L189
	.long	.L202-.L189
	.long	.L202-.L189
	.long	.L192-.L189
	.long	.L202-.L189
	.long	.L191-.L189
	.long	.L203-.L189
	.long	.L188-.L189
	.text
.L188:
	movq	$2, -8232(%rbp)
	jmp	.L196
.L192:
	leaq	-4112(%rbp), %rdx
	leaq	-8208(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	cria_vector
	leaq	-8208(%rbp), %rax
	movl	$2, %edx
	movl	$1024, %esi
	movq	%rax, %rdi
	call	shell_sort
	movq	count_comparacoes(%rip), %rax
	movl	%eax, %ecx
	movl	-8244(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	vector_das_comparacoes(%rip), %rax
	movl	%ecx, (%rdx,%rax)
	movq	$0, count_comparacoes(%rip)
	addl	$1, -8244(%rbp)
	movq	$7, -8232(%rbp)
	jmp	.L196
.L194:
	leaq	vector_das_comparacoes(%rip), %rax
	movq	%rax, %rdi
	call	media
	movl	%eax, -8240(%rbp)
	movl	-8240(%rbp), %eax
	cltq
	movq	%rax, -8224(%rbp)
	leaq	vector_das_comparacoes(%rip), %rax
	movq	%rax, %rdi
	call	desvio_padrao
	movl	%eax, -8236(%rbp)
	movl	-8236(%rbp), %eax
	cltq
	movq	%rax, -8216(%rbp)
	movq	-8224(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-8216(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -8232(%rbp)
	jmp	.L196
.L191:
	cmpl	$999, -8244(%rbp)
	jg	.L197
	movq	$5, -8232(%rbp)
	jmp	.L196
.L197:
	movq	$0, -8232(%rbp)
	jmp	.L196
.L193:
	movq	$0, count_comparacoes(%rip)
	movl	$0, -8244(%rbp)
	movq	$7, -8232(%rbp)
	jmp	.L196
.L202:
	nop
.L196:
	jmp	.L199
.L203:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L201
	call	__stack_chk_fail@PLT
.L201:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	mil_shell_sort_knuth, .-mil_shell_sort_knuth
	.globl	raiz_quadrada
	.type	raiz_quadrada, @function
raiz_quadrada:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$8, -8(%rbp)
.L219:
	cmpq	$9, -8(%rbp)
	ja	.L220
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L207(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L207(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L207:
	.long	.L212-.L207
	.long	.L220-.L207
	.long	.L220-.L207
	.long	.L211-.L207
	.long	.L220-.L207
	.long	.L210-.L207
	.long	.L220-.L207
	.long	.L209-.L207
	.long	.L208-.L207
	.long	.L206-.L207
	.text
.L208:
	cmpl	$0, -20(%rbp)
	jg	.L213
	movq	$3, -8(%rbp)
	jmp	.L215
.L213:
	movq	$0, -8(%rbp)
	jmp	.L215
.L211:
	movl	$0, %eax
	jmp	.L216
.L206:
	movl	-12(%rbp), %eax
	cmpl	-16(%rbp), %eax
	jge	.L217
	movq	$7, -8(%rbp)
	jmp	.L215
.L217:
	movq	$5, -8(%rbp)
	jmp	.L215
.L210:
	movl	-16(%rbp), %eax
	jmp	.L216
.L212:
	movl	-20(%rbp), %eax
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	movl	%eax, -12(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L215
.L209:
	movl	-12(%rbp), %eax
	movl	%eax, -16(%rbp)
	movl	-20(%rbp), %eax
	cltd
	idivl	-16(%rbp)
	movl	%eax, %edx
	movl	-16(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	movl	%eax, -12(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L215
.L220:
	nop
.L215:
	jmp	.L219
.L216:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	raiz_quadrada, .-raiz_quadrada
	.section	.rodata
	.align 8
.LC10:
	.string	"M\303\251dia do Quick Sort mediano: %lld\n"
	.align 8
.LC11:
	.string	"Desvio padrao do Quick Sort mediano: %lld\n"
	.text
	.globl	mil_quick_sort_mediano
	.type	mil_quick_sort_mediano, @function
mil_quick_sort_mediano:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$64, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$3, -8232(%rbp)
.L234:
	cmpq	$8, -8232(%rbp)
	ja	.L237
	movq	-8232(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L224(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L224(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L224:
	.long	.L229-.L224
	.long	.L228-.L224
	.long	.L237-.L224
	.long	.L227-.L224
	.long	.L226-.L224
	.long	.L225-.L224
	.long	.L237-.L224
	.long	.L237-.L224
	.long	.L238-.L224
	.text
.L226:
	cmpl	$999, -8244(%rbp)
	jg	.L230
	movq	$0, -8232(%rbp)
	jmp	.L232
.L230:
	movq	$5, -8232(%rbp)
	jmp	.L232
.L228:
	movq	$0, count_comparacoes(%rip)
	movl	$0, -8244(%rbp)
	movq	$4, -8232(%rbp)
	jmp	.L232
.L227:
	movq	$1, -8232(%rbp)
	jmp	.L232
.L225:
	leaq	vector_das_comparacoes(%rip), %rax
	movq	%rax, %rdi
	call	media
	movl	%eax, -8240(%rbp)
	movl	-8240(%rbp), %eax
	cltq
	movq	%rax, -8224(%rbp)
	leaq	vector_das_comparacoes(%rip), %rax
	movq	%rax, %rdi
	call	desvio_padrao
	movl	%eax, -8236(%rbp)
	movl	-8236(%rbp), %eax
	cltq
	movq	%rax, -8216(%rbp)
	movq	-8224(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-8216(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -8232(%rbp)
	jmp	.L232
.L229:
	leaq	-4112(%rbp), %rdx
	leaq	-8208(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	cria_vector
	leaq	-8208(%rbp), %rax
	movl	$2, %ecx
	movl	$1023, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	quick_sort
	movq	count_comparacoes(%rip), %rax
	movl	%eax, %ecx
	movl	-8244(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	vector_das_comparacoes(%rip), %rax
	movl	%ecx, (%rdx,%rax)
	movq	$0, count_comparacoes(%rip)
	addl	$1, -8244(%rbp)
	movq	$4, -8232(%rbp)
	jmp	.L232
.L237:
	nop
.L232:
	jmp	.L234
.L238:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L236
	call	__stack_chk_fail@PLT
.L236:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	mil_quick_sort_mediano, .-mil_quick_sort_mediano
	.globl	particao
	.type	particao, @function
particao:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movl	%edx, -48(%rbp)
	movl	%ecx, -52(%rbp)
	movq	$10, -8(%rbp)
.L274:
	cmpq	$23, -8(%rbp)
	ja	.L276
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L242(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L242(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L242:
	.long	.L259-.L242
	.long	.L276-.L242
	.long	.L276-.L242
	.long	.L276-.L242
	.long	.L258-.L242
	.long	.L257-.L242
	.long	.L256-.L242
	.long	.L255-.L242
	.long	.L254-.L242
	.long	.L253-.L242
	.long	.L252-.L242
	.long	.L276-.L242
	.long	.L251-.L242
	.long	.L250-.L242
	.long	.L249-.L242
	.long	.L276-.L242
	.long	.L248-.L242
	.long	.L247-.L242
	.long	.L246-.L242
	.long	.L276-.L242
	.long	.L245-.L242
	.long	.L244-.L242
	.long	.L243-.L242
	.long	.L241-.L242
	.text
.L246:
	cmpl	$2, -52(%rbp)
	jne	.L260
	movq	$22, -8(%rbp)
	jmp	.L262
.L260:
	movq	$8, -8(%rbp)
	jmp	.L262
.L258:
	movl	-20(%rbp), %eax
	movl	%eax, -24(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L262
.L249:
	movl	-12(%rbp), %eax
	cmpl	-48(%rbp), %eax
	jge	.L263
	movq	$13, -8(%rbp)
	jmp	.L262
.L263:
	movq	$21, -8(%rbp)
	jmp	.L262
.L251:
	movl	-44(%rbp), %eax
	movl	%eax, -24(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L262
.L254:
	movl	-44(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -16(%rbp)
	movl	-44(%rbp), %eax
	movl	%eax, -12(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L262
.L241:
	addl	$1, -16(%rbp)
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-40(%rbp), %rax
	addq	%rcx, %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	troca
	movq	count_trocas(%rip), %rax
	addq	$1, %rax
	movq	%rax, count_trocas(%rip)
	movq	$20, -8(%rbp)
	jmp	.L262
.L248:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -28(%rbp)
	jne	.L265
	movq	$4, -8(%rbp)
	jmp	.L262
.L265:
	movq	$17, -8(%rbp)
	jmp	.L262
.L244:
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movl	-16(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,4), %rcx
	movq	-40(%rbp), %rax
	addq	%rcx, %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	troca
	movq	count_trocas(%rip), %rax
	addq	$1, %rax
	movq	%rax, count_trocas(%rip)
	movq	$9, -8(%rbp)
	jmp	.L262
.L253:
	movl	-16(%rbp), %eax
	addl	$1, %eax
	jmp	.L275
.L250:
	movq	count_comparacoes(%rip), %rax
	addq	$1, %rax
	movq	%rax, count_comparacoes(%rip)
	movq	$5, -8(%rbp)
	jmp	.L262
.L247:
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movl	-24(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-40(%rbp), %rax
	addq	%rcx, %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	troca
	movq	$8, -8(%rbp)
	jmp	.L262
.L256:
	movl	-44(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -28(%rbp)
	jne	.L268
	movq	$12, -8(%rbp)
	jmp	.L262
.L268:
	movq	$16, -8(%rbp)
	jmp	.L262
.L243:
	movl	-48(%rbp), %eax
	subl	-44(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	movl	%eax, %edx
	movl	-44(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -20(%rbp)
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-40(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %ecx
	movl	-44(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rsi
	movq	-40(%rbp), %rax
	addq	%rsi, %rax
	movl	(%rax), %eax
	movl	%ecx, %esi
	movl	%eax, %edi
	call	mediano
	movl	%eax, -28(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L262
.L257:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -28(%rbp)
	jl	.L270
	movq	$23, -8(%rbp)
	jmp	.L262
.L270:
	movq	$20, -8(%rbp)
	jmp	.L262
.L252:
	movl	-48(%rbp), %eax
	movl	%eax, -24(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L262
.L259:
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -28(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L262
.L255:
	cmpl	$1, -52(%rbp)
	jne	.L272
	movq	$0, -8(%rbp)
	jmp	.L262
.L272:
	movq	$18, -8(%rbp)
	jmp	.L262
.L245:
	addl	$1, -12(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L262
.L276:
	nop
.L262:
	jmp	.L274
.L275:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	particao, .-particao
	.section	.rodata
	.align 8
.LC12:
	.string	"M\303\251dia do Quick Sort \303\272ltimo elemento: %lld\n"
	.align 8
.LC13:
	.string	"Desvio padrao do Quick Sort \303\272ltimo elemento: %lld\n"
	.text
	.globl	mil_quick_sort_ultimoelemento
	.type	mil_quick_sort_ultimoelemento, @function
mil_quick_sort_ultimoelemento:
.LFB14:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$64, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$5, -8232(%rbp)
.L290:
	cmpq	$8, -8232(%rbp)
	ja	.L293
	movq	-8232(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L280(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L280(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L280:
	.long	.L293-.L280
	.long	.L285-.L280
	.long	.L284-.L280
	.long	.L293-.L280
	.long	.L293-.L280
	.long	.L283-.L280
	.long	.L282-.L280
	.long	.L294-.L280
	.long	.L279-.L280
	.text
.L279:
	leaq	-4112(%rbp), %rdx
	leaq	-8208(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	cria_vector
	leaq	-8208(%rbp), %rax
	movl	$1, %ecx
	movl	$1023, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	quick_sort
	movq	count_comparacoes(%rip), %rax
	movl	%eax, %ecx
	movl	-8244(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	vector_das_comparacoes(%rip), %rax
	movl	%ecx, (%rdx,%rax)
	movq	$0, count_comparacoes(%rip)
	addl	$1, -8244(%rbp)
	movq	$2, -8232(%rbp)
	jmp	.L286
.L285:
	movq	$0, count_comparacoes(%rip)
	movl	$0, -8244(%rbp)
	movq	$2, -8232(%rbp)
	jmp	.L286
.L282:
	leaq	vector_das_comparacoes(%rip), %rax
	movq	%rax, %rdi
	call	media
	movl	%eax, -8240(%rbp)
	movl	-8240(%rbp), %eax
	cltq
	movq	%rax, -8224(%rbp)
	leaq	vector_das_comparacoes(%rip), %rax
	movq	%rax, %rdi
	call	desvio_padrao
	movl	%eax, -8236(%rbp)
	movl	-8236(%rbp), %eax
	cltq
	movq	%rax, -8216(%rbp)
	movq	-8224(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-8216(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -8232(%rbp)
	jmp	.L286
.L283:
	movq	$1, -8232(%rbp)
	jmp	.L286
.L284:
	cmpl	$999, -8244(%rbp)
	jg	.L288
	movq	$8, -8232(%rbp)
	jmp	.L286
.L288:
	movq	$6, -8232(%rbp)
	jmp	.L286
.L293:
	nop
.L286:
	jmp	.L290
.L294:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L292
	call	__stack_chk_fail@PLT
.L292:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE14:
	.size	mil_quick_sort_ultimoelemento, .-mil_quick_sort_ultimoelemento
	.globl	cria_vector
	.type	cria_vector, @function
cria_vector:
.LFB15:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	$6, -16(%rbp)
.L304:
	cmpq	$6, -16(%rbp)
	je	.L296
	cmpq	$6, -16(%rbp)
	ja	.L306
	cmpq	$5, -16(%rbp)
	je	.L298
	cmpq	$5, -16(%rbp)
	ja	.L306
	cmpq	$2, -16(%rbp)
	je	.L299
	cmpq	$3, -16(%rbp)
	jne	.L306
	jmp	.L305
.L296:
	movl	$0, -20(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L301
.L298:
	movl	$2048, %esi
	movl	$0, %edi
	call	aleat
	movq	%rax, -8(%rbp)
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	-8(%rbp), %rdx
	movl	%edx, (%rax)
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	-20(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-48(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	addl	$1, -20(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L301
.L299:
	cmpl	$1023, -20(%rbp)
	jg	.L302
	movq	$5, -16(%rbp)
	jmp	.L301
.L302:
	movq	$3, -16(%rbp)
	jmp	.L301
.L306:
	nop
.L301:
	jmp	.L304
.L305:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE15:
	.size	cria_vector, .-cria_vector
	.globl	media
	.type	media, @function
media:
.LFB17:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movq	$4, -8(%rbp)
.L320:
	cmpq	$6, -8(%rbp)
	ja	.L322
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L310(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L310(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L310:
	.long	.L315-.L310
	.long	.L314-.L310
	.long	.L313-.L310
	.long	.L312-.L310
	.long	.L311-.L310
	.long	.L322-.L310
	.long	.L309-.L310
	.text
.L311:
	movq	$6, -8(%rbp)
	jmp	.L316
.L314:
	cmpl	$999, -28(%rbp)
	jg	.L317
	movq	$2, -8(%rbp)
	jmp	.L316
.L317:
	movq	$3, -8(%rbp)
	jmp	.L316
.L312:
	movq	-24(%rbp), %rcx
	movabsq	$2361183241434822607, %rdx
	movq	%rcx, %rax
	imulq	%rdx
	movq	%rdx, %rax
	sarq	$7, %rax
	sarq	$63, %rcx
	movq	%rcx, %rdx
	subq	%rdx, %rax
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L316
.L309:
	movq	$0, -24(%rbp)
	movl	$0, -28(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L316
.L315:
	movq	-16(%rbp), %rax
	jmp	.L321
.L313:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cltq
	addq	%rax, -24(%rbp)
	addl	$1, -28(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L316
.L322:
	nop
.L316:
	jmp	.L320
.L321:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE17:
	.size	media, .-media
	.globl	troca
	.type	troca, @function
troca:
.LFB19:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$2, -8(%rbp)
.L329:
	cmpq	$2, -8(%rbp)
	je	.L324
	cmpq	$2, -8(%rbp)
	ja	.L330
	cmpq	$0, -8(%rbp)
	je	.L331
	cmpq	$1, -8(%rbp)
	jne	.L330
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -12(%rbp)
	movq	-32(%rbp), %rax
	movl	(%rax), %edx
	movq	-24(%rbp), %rax
	movl	%edx, (%rax)
	movq	-32(%rbp), %rax
	movl	-12(%rbp), %edx
	movl	%edx, (%rax)
	movq	$0, -8(%rbp)
	jmp	.L327
.L324:
	movq	$1, -8(%rbp)
	jmp	.L327
.L330:
	nop
.L327:
	jmp	.L329
.L331:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE19:
	.size	troca, .-troca
	.section	.rodata
	.align 8
.LC14:
	.string	"M\303\251dia da Pesquisa Sequencial: %lld\n"
	.align 8
.LC15:
	.string	"Desvio padr\303\243o da Pesquisa Sequencial: %lld\n"
	.text
	.globl	mil_pesquisa_sequencial
	.type	mil_pesquisa_sequencial, @function
mil_pesquisa_sequencial:
.LFB20:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$64, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$6, -8240(%rbp)
.L345:
	cmpq	$8, -8240(%rbp)
	ja	.L348
	movq	-8240(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L335(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L335(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L335:
	.long	.L340-.L335
	.long	.L348-.L335
	.long	.L339-.L335
	.long	.L348-.L335
	.long	.L338-.L335
	.long	.L337-.L335
	.long	.L336-.L335
	.long	.L348-.L335
	.long	.L349-.L335
	.text
.L338:
	cmpl	$999, -8256(%rbp)
	jg	.L341
	movq	$0, -8240(%rbp)
	jmp	.L343
.L341:
	movq	$5, -8240(%rbp)
	jmp	.L343
.L336:
	movq	$2, -8240(%rbp)
	jmp	.L343
.L337:
	leaq	vector_das_comparacoes(%rip), %rax
	movq	%rax, %rdi
	call	media
	movl	%eax, -8252(%rbp)
	movl	-8252(%rbp), %eax
	cltq
	movq	%rax, -8232(%rbp)
	leaq	vector_das_comparacoes(%rip), %rax
	movq	%rax, %rdi
	call	desvio_padrao
	movl	%eax, -8248(%rbp)
	movl	-8248(%rbp), %eax
	cltq
	movq	%rax, -8224(%rbp)
	movq	-8232(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-8224(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -8240(%rbp)
	jmp	.L343
.L340:
	leaq	-4112(%rbp), %rdx
	leaq	-8208(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	cria_vector
	movl	$2048, %esi
	movl	$0, %edi
	call	aleat
	movq	%rax, -8216(%rbp)
	movq	-8216(%rbp), %rax
	movl	%eax, -8244(%rbp)
	movl	-8244(%rbp), %edx
	leaq	-4112(%rbp), %rax
	movl	%edx, %ecx
	movl	$1024, %edx
	movl	$2, %esi
	movq	%rax, %rdi
	call	pesquisa_sequencial
	movq	count_comparacoes(%rip), %rax
	movl	%eax, %ecx
	movl	-8256(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	vector_das_comparacoes(%rip), %rax
	movl	%ecx, (%rdx,%rax)
	movq	$0, count_comparacoes(%rip)
	addl	$1, -8256(%rbp)
	movq	$4, -8240(%rbp)
	jmp	.L343
.L339:
	movq	$0, count_comparacoes(%rip)
	movl	$0, -8256(%rbp)
	movq	$4, -8240(%rbp)
	jmp	.L343
.L348:
	nop
.L343:
	jmp	.L345
.L349:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L347
	call	__stack_chk_fail@PLT
.L347:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE20:
	.size	mil_pesquisa_sequencial, .-mil_pesquisa_sequencial
	.globl	aleat
	.type	aleat, @function
aleat:
.LFB21:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$1, -8(%rbp)
.L355:
	cmpq	$0, -8(%rbp)
	je	.L351
	cmpq	$1, -8(%rbp)
	jne	.L357
	call	rand@PLT
	movl	%eax, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L353
.L351:
	movl	-12(%rbp), %eax
	cltq
	movq	-32(%rbp), %rdx
	subq	-24(%rbp), %rdx
	leaq	1(%rdx), %rcx
	cqto
	idivq	%rcx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	jmp	.L356
.L357:
	nop
.L353:
	jmp	.L355
.L356:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE21:
	.size	aleat, .-aleat
	.globl	pesquisa_binaria
	.type	pesquisa_binaria, @function
pesquisa_binaria:
.LFB22:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movl	%edx, -48(%rbp)
	movl	%ecx, -52(%rbp)
	movq	$9, -8(%rbp)
.L379:
	cmpq	$12, -8(%rbp)
	ja	.L380
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L361(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L361(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L361:
	.long	.L380-.L361
	.long	.L380-.L361
	.long	.L370-.L361
	.long	.L369-.L361
	.long	.L368-.L361
	.long	.L367-.L361
	.long	.L366-.L361
	.long	.L365-.L361
	.long	.L380-.L361
	.long	.L364-.L361
	.long	.L363-.L361
	.long	.L362-.L361
	.long	.L360-.L361
	.text
.L368:
	movl	-12(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -20(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L371
.L360:
	movl	$0, -20(%rbp)
	movl	-48(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -16(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L371
.L369:
	movl	-12(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -16(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L371
.L362:
	movl	-16(%rbp), %eax
	subl	-20(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	movl	%eax, %edx
	movl	-20(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -12(%rbp)
	movq	count_comparacoes(%rip), %rax
	addq	$1, %rax
	movq	%rax, count_comparacoes(%rip)
	movq	$7, -8(%rbp)
	jmp	.L371
.L364:
	movq	$12, -8(%rbp)
	jmp	.L371
.L366:
	movl	$-1, %eax
	jmp	.L372
.L367:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -52(%rbp)
	jle	.L373
	movq	$4, -8(%rbp)
	jmp	.L371
.L373:
	movq	$3, -8(%rbp)
	jmp	.L371
.L363:
	movl	-12(%rbp), %eax
	jmp	.L372
.L365:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -52(%rbp)
	jne	.L375
	movq	$10, -8(%rbp)
	jmp	.L371
.L375:
	movq	$5, -8(%rbp)
	jmp	.L371
.L370:
	movl	-20(%rbp), %eax
	cmpl	-16(%rbp), %eax
	jg	.L377
	movq	$11, -8(%rbp)
	jmp	.L371
.L377:
	movq	$6, -8(%rbp)
	jmp	.L371
.L380:
	nop
.L371:
	jmp	.L379
.L372:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE22:
	.size	pesquisa_binaria, .-pesquisa_binaria
	.globl	mediano
	.type	mediano, @function
mediano:
.LFB23:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movl	%edx, -28(%rbp)
	movq	$13, -8(%rbp)
.L416:
	cmpq	$17, -8(%rbp)
	ja	.L417
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L384(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L384(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L384:
	.long	.L397-.L384
	.long	.L396-.L384
	.long	.L395-.L384
	.long	.L394-.L384
	.long	.L417-.L384
	.long	.L393-.L384
	.long	.L417-.L384
	.long	.L392-.L384
	.long	.L391-.L384
	.long	.L390-.L384
	.long	.L389-.L384
	.long	.L417-.L384
	.long	.L417-.L384
	.long	.L388-.L384
	.long	.L387-.L384
	.long	.L386-.L384
	.long	.L385-.L384
	.long	.L383-.L384
	.text
.L387:
	movl	-20(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L398
	movq	$5, -8(%rbp)
	jmp	.L400
.L398:
	movq	$0, -8(%rbp)
	jmp	.L400
.L386:
	movl	-24(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jge	.L401
	movq	$17, -8(%rbp)
	jmp	.L400
.L401:
	movq	$10, -8(%rbp)
	jmp	.L400
.L391:
	movl	-24(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jle	.L403
	movq	$3, -8(%rbp)
	jmp	.L400
.L403:
	movq	$15, -8(%rbp)
	jmp	.L400
.L396:
	movl	-24(%rbp), %eax
	jmp	.L405
.L394:
	movl	-24(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L406
	movq	$1, -8(%rbp)
	jmp	.L400
.L406:
	movq	$15, -8(%rbp)
	jmp	.L400
.L385:
	movl	-20(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jle	.L408
	movq	$2, -8(%rbp)
	jmp	.L400
.L408:
	movq	$8, -8(%rbp)
	jmp	.L400
.L390:
	movl	-24(%rbp), %eax
	jmp	.L405
.L388:
	movl	-20(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jle	.L410
	movq	$14, -8(%rbp)
	jmp	.L400
.L410:
	movq	$0, -8(%rbp)
	jmp	.L400
.L383:
	movl	-24(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jle	.L412
	movq	$9, -8(%rbp)
	jmp	.L400
.L412:
	movq	$7, -8(%rbp)
	jmp	.L400
.L393:
	movl	-20(%rbp), %eax
	jmp	.L405
.L389:
	movl	-28(%rbp), %eax
	jmp	.L405
.L397:
	movl	-20(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jge	.L414
	movq	$16, -8(%rbp)
	jmp	.L400
.L414:
	movq	$8, -8(%rbp)
	jmp	.L400
.L392:
	movl	-28(%rbp), %eax
	jmp	.L405
.L395:
	movl	-20(%rbp), %eax
	jmp	.L405
.L417:
	nop
.L400:
	jmp	.L416
.L405:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE23:
	.size	mediano, .-mediano
	.section	.rodata
	.align 8
.LC16:
	.string	"ATEN\303\207\303\203O, SE VOC\303\212 N\303\203O ORDENOU O VETOR ANTERIORMENTE N\303\203O DAR\303\201 CERTO A PESQUISA BIN\303\201RIA."
	.align 8
.LC17:
	.string	"Se voc\303\252 ordenou e deseja seguir, aperte 1.\nCaso n\303\243o ordenou, aperte qualquer tecla e escolha uma op\303\247\303\243o para ordenar."
.LC18:
	.string	"%d"
	.align 8
.LC19:
	.string	"Elemento %d n\303\243o encontrado no vetor.\n"
	.align 8
.LC20:
	.string	"-------------------------------------------------------------"
.LC21:
	.string	"# MENU DO MEU ALGORITMO: "
	.align 8
.LC22:
	.string	"ATEN\303\207\303\203O: N\303\203O APERTE NENHUM OUTRO BOT\303\203O N\303\203O PEDIDO AO LONGO DO PROGRAMA."
	.align 8
.LC23:
	.string	"Se voc\303\252 apertar, REINICIE O PROGRAMA."
.LC24:
	.string	"1 - Cria um novo vetor."
	.align 8
.LC25:
	.string	"2 - Imprime 100 primeiros elementos."
	.align 8
.LC26:
	.string	"3 - Ordena vetor Selection Sort."
.LC27:
	.string	"4 - Ordena vetor Quick Sort."
.LC28:
	.string	"        1 - \303\232ltimo elemento."
	.align 8
.LC29:
	.string	"        2 - Elemento mediano entre o primeiro, o meio e o \303\272ltimo elemento do vetor."
.LC30:
	.string	"5 - Ordena vetor Shell Sort."
	.align 8
.LC31:
	.string	"        1. Espa\303\247amento Padr\303\243o divindo por 2."
	.align 8
.LC32:
	.string	"        2. Espa\303\247amento de Knuth (3\303\227gap+1 at\303\251 ser menor que n/3)."
.LC33:
	.string	"6 - Pesquisa Sequencial:"
.LC34:
	.string	"        1 - Usu\303\241rio Escolhe."
	.align 8
.LC35:
	.string	"        2 - Gerado Aleat\303\263riamente."
.LC36:
	.string	"7 - Pesquisa Bin\303\241ria:"
	.align 8
.LC37:
	.string	"8 - Ordena\303\247\303\265es e Buscas de TODOS 1000 vezes."
	.align 8
.LC38:
	.string	"9 - Caso deseje encerrar o programa."
.LC39:
	.string	"Digite a Entrada: "
	.align 8
.LC40:
	.string	"N\303\272mero de Compara\303\247\303\265es feitas: %lld\n"
	.align 8
.LC41:
	.string	"Elemento %d encontrado no \303\255ndice %d do vetor.\n"
	.align 8
.LC42:
	.string	"Escolha a forma de escolher o espa\303\247amento: "
	.align 8
.LC43:
	.string	"1. Espa\303\247amento Padr\303\243o divindo por 2."
	.align 8
.LC44:
	.string	"2. Espa\303\247amento de Knuth (3\303\227gap+1 at\303\251 ser menor que n/3)."
	.align 8
.LC45:
	.string	"N\303\272mero de Compara\303\247\303\265es: %lld\n"
.LC46:
	.string	"N\303\272mero de Trocas: %lld\n"
	.align 8
.LC47:
	.string	"Escolha a forma de se fazer a pesquisa: "
	.align 8
.LC48:
	.string	"1. Elemento que usu\303\241rio escolhe."
	.align 8
.LC49:
	.string	"2. Elemento gerado aleat\303\263riamente"
	.align 8
.LC50:
	.string	"Escolha um elemento para ser pesquisado: "
	.align 8
.LC51:
	.string	"Escolha a forma de escolher o piv\303\264: "
	.align 8
.LC52:
	.string	"1. Escolhendo o \303\272ltimo elemento."
	.align 8
.LC53:
	.string	"2. Pegando 3 elementos e escolhendo o do meio."
	.align 8
.LC54:
	.string	"Vetor Criado, para exibir uma parte dele, digite 2."
	.align 8
.LC55:
	.string	"--------------------------------------------------"
	.align 8
.LC56:
	.string	"# M\303\211DIAS E DESVIOS PADR\303\225ES DAS COMPARA\303\207\303\225ES DOS ALGORITMOS: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB24:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$4096, %rsp
	orq	$0, (%rsp)
	addq	$-128, %rsp
	movl	%edi, -8292(%rbp)
	movq	%rsi, -8304(%rbp)
	movq	%rdx, -8312(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, -8256(%rbp)
	jmp	.L419
.L420:
	movl	-8256(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	vector_das_comparacoes(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -8256(%rbp)
.L419:
	cmpl	$999, -8256(%rbp)
	jle	.L420
	nop
.L421:
	movq	$0, count_trocas(%rip)
	nop
.L422:
	movq	$0, count_comparacoes(%rip)
	nop
.L423:
	movq	$0, _TIG_IZ_0hIZ_envp(%rip)
	nop
.L424:
	movq	$0, _TIG_IZ_0hIZ_argv(%rip)
	nop
.L425:
	movl	$0, _TIG_IZ_0hIZ_argc(%rip)
	nop
	nop
.L426:
.L427:
#APP
# 455 "Antonio-RF_Sorting-and-Searching-Algorithms_tp.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-0hIZ--0
# 0 "" 2
#NO_APP
	movl	-8292(%rbp), %eax
	movl	%eax, _TIG_IZ_0hIZ_argc(%rip)
	movq	-8304(%rbp), %rax
	movq	%rax, _TIG_IZ_0hIZ_argv(%rip)
	movq	-8312(%rbp), %rax
	movq	%rax, _TIG_IZ_0hIZ_envp(%rip)
	nop
	movq	$40, -8232(%rbp)
.L507:
	cmpq	$57, -8232(%rbp)
	ja	.L510
	movq	-8232(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L430(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L430(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L430:
	.long	.L510-.L430
	.long	.L470-.L430
	.long	.L469-.L430
	.long	.L468-.L430
	.long	.L467-.L430
	.long	.L466-.L430
	.long	.L465-.L430
	.long	.L464-.L430
	.long	.L463-.L430
	.long	.L462-.L430
	.long	.L461-.L430
	.long	.L460-.L430
	.long	.L459-.L430
	.long	.L458-.L430
	.long	.L457-.L430
	.long	.L456-.L430
	.long	.L455-.L430
	.long	.L510-.L430
	.long	.L454-.L430
	.long	.L453-.L430
	.long	.L452-.L430
	.long	.L451-.L430
	.long	.L510-.L430
	.long	.L450-.L430
	.long	.L510-.L430
	.long	.L510-.L430
	.long	.L449-.L430
	.long	.L510-.L430
	.long	.L510-.L430
	.long	.L510-.L430
	.long	.L448-.L430
	.long	.L510-.L430
	.long	.L447-.L430
	.long	.L446-.L430
	.long	.L445-.L430
	.long	.L510-.L430
	.long	.L444-.L430
	.long	.L443-.L430
	.long	.L510-.L430
	.long	.L510-.L430
	.long	.L442-.L430
	.long	.L441-.L430
	.long	.L510-.L430
	.long	.L440-.L430
	.long	.L439-.L430
	.long	.L438-.L430
	.long	.L437-.L430
	.long	.L510-.L430
	.long	.L436-.L430
	.long	.L435-.L430
	.long	.L434-.L430
	.long	.L433-.L430
	.long	.L510-.L430
	.long	.L510-.L430
	.long	.L432-.L430
	.long	.L431-.L430
	.long	.L510-.L430
	.long	.L429-.L430
	.text
.L454:
	cmpl	$-1, -8244(%rbp)
	je	.L471
	movq	$45, -8232(%rbp)
	jmp	.L473
.L471:
	movq	$21, -8232(%rbp)
	jmp	.L473
.L434:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-8268(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$2, -8232(%rbp)
	jmp	.L473
.L435:
	movl	-8272(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$37, -8232(%rbp)
	jmp	.L473
.L467:
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$10, %edi
	call	putchar@PLT
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$10, %edi
	call	putchar@PLT
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC25(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC26(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC27(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC29(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC31(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC32(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC33(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC34(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC35(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC36(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC34(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC35(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC37(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC38(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$10, %edi
	call	putchar@PLT
	leaq	.LC39(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -8252(%rbp)
	movq	$12, -8232(%rbp)
	jmp	.L473
.L448:
	movl	-8272(%rbp), %edx
	movl	-8276(%rbp), %esi
	leaq	-4112(%rbp), %rax
	movl	%edx, %ecx
	movl	$1024, %edx
	movq	%rax, %rdi
	call	pesquisa_sequencial
	movl	%eax, -8240(%rbp)
	movl	-8240(%rbp), %eax
	movl	%eax, -8248(%rbp)
	movq	count_comparacoes(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC40(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, count_comparacoes(%rip)
	movq	$43, -8232(%rbp)
	jmp	.L473
.L457:
	movl	-8276(%rbp), %eax
	cmpl	$1, %eax
	jne	.L474
	movq	$20, -8232(%rbp)
	jmp	.L473
.L474:
	movq	$3, -8232(%rbp)
	jmp	.L473
.L456:
	movl	-8288(%rbp), %eax
	cmpl	$5, %eax
	jne	.L476
	movq	$36, -8232(%rbp)
	jmp	.L473
.L476:
	movq	$8, -8232(%rbp)
	jmp	.L473
.L459:
	cmpl	$999, -8252(%rbp)
	jg	.L478
	movq	$44, -8232(%rbp)
	jmp	.L473
.L478:
	movq	$1, -8232(%rbp)
	jmp	.L473
.L463:
	movl	-8288(%rbp), %eax
	cmpl	$6, %eax
	jne	.L480
	movq	$19, -8232(%rbp)
	jmp	.L473
.L480:
	movq	$6, -8232(%rbp)
	jmp	.L473
.L438:
	movl	-8260(%rbp), %eax
	movl	-8244(%rbp), %edx
	movl	%eax, %esi
	leaq	.LC41(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$37, -8232(%rbp)
	jmp	.L473
.L432:
	movl	-8288(%rbp), %eax
	cmpl	$2, %eax
	jne	.L482
	movq	$23, -8232(%rbp)
	jmp	.L473
.L482:
	movq	$32, -8232(%rbp)
	jmp	.L473
.L470:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L508
	jmp	.L509
.L450:
	leaq	-8208(%rbp), %rax
	movq	%rax, %rdi
	call	imprime_parte
	movq	$37, -8232(%rbp)
	jmp	.L473
.L468:
	movl	-8276(%rbp), %eax
	cmpl	$2, %eax
	jne	.L485
	movq	$9, -8232(%rbp)
	jmp	.L473
.L485:
	movq	$30, -8232(%rbp)
	jmp	.L473
.L455:
	movl	-8264(%rbp), %eax
	cmpl	$2, %eax
	jne	.L487
	movq	$46, -8232(%rbp)
	jmp	.L473
.L487:
	movq	$11, -8232(%rbp)
	jmp	.L473
.L451:
	movl	-8260(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$37, -8232(%rbp)
	jmp	.L473
.L444:
	leaq	.LC42(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC43(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC44(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-8280(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-8280(%rbp), %edx
	leaq	-8208(%rbp), %rax
	movl	$1024, %esi
	movq	%rax, %rdi
	call	shell_sort
	movq	count_comparacoes(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC45(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	count_trocas(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC46(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-8208(%rbp), %rax
	movq	%rax, %rdi
	call	imprime_parte
	movq	$0, count_comparacoes(%rip)
	movq	$0, count_trocas(%rip)
	movq	$37, -8232(%rbp)
	jmp	.L473
.L429:
	leaq	.LC47(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC48(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC49(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-8264(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -8260(%rbp)
	movq	$55, -8232(%rbp)
	jmp	.L473
.L449:
	movl	-8272(%rbp), %eax
	movl	-8248(%rbp), %edx
	movl	%eax, %esi
	leaq	.LC41(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$37, -8232(%rbp)
	jmp	.L473
.L460:
	movl	-8260(%rbp), %edx
	movl	-8264(%rbp), %esi
	leaq	-8208(%rbp), %rax
	movl	%edx, %ecx
	movl	$1024, %edx
	movq	%rax, %rdi
	call	pesquisa_binaria
	movl	%eax, -8236(%rbp)
	movl	-8236(%rbp), %eax
	movl	%eax, -8244(%rbp)
	movq	count_comparacoes(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC40(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, count_comparacoes(%rip)
	movq	$18, -8232(%rbp)
	jmp	.L473
.L462:
	movl	$2048, %esi
	movl	$0, %edi
	call	aleat
	movq	%rax, -8216(%rbp)
	movq	-8216(%rbp), %rax
	movl	%eax, -8272(%rbp)
	movq	$30, -8232(%rbp)
	jmp	.L473
.L458:
	leaq	.LC50(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-8260(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$16, -8232(%rbp)
	jmp	.L473
.L433:
	leaq	.LC51(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC52(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC53(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-8284(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-8284(%rbp), %edx
	leaq	-8208(%rbp), %rax
	movl	%edx, %ecx
	movl	$1023, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	quick_sort
	movq	count_comparacoes(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC45(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	count_trocas(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC46(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-8208(%rbp), %rax
	movq	%rax, %rdi
	call	imprime_parte
	movq	$0, count_comparacoes(%rip)
	movq	$0, count_trocas(%rip)
	movq	$37, -8232(%rbp)
	jmp	.L473
.L453:
	leaq	.LC47(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC48(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC49(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-8276(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -8272(%rbp)
	movq	$14, -8232(%rbp)
	jmp	.L473
.L447:
	movl	-8288(%rbp), %eax
	cmpl	$3, %eax
	jne	.L489
	movq	$7, -8232(%rbp)
	jmp	.L473
.L489:
	movq	$34, -8232(%rbp)
	jmp	.L473
.L442:
	movq	$4, -8232(%rbp)
	jmp	.L473
.L431:
	movl	-8264(%rbp), %eax
	cmpl	$1, %eax
	jne	.L491
	movq	$13, -8232(%rbp)
	jmp	.L473
.L491:
	movq	$16, -8232(%rbp)
	jmp	.L473
.L465:
	movl	-8288(%rbp), %eax
	cmpl	$7, %eax
	jne	.L493
	movq	$50, -8232(%rbp)
	jmp	.L473
.L493:
	movq	$33, -8232(%rbp)
	jmp	.L473
.L445:
	movl	-8288(%rbp), %eax
	cmpl	$4, %eax
	jne	.L495
	movq	$51, -8232(%rbp)
	jmp	.L473
.L495:
	movq	$15, -8232(%rbp)
	jmp	.L473
.L436:
	leaq	-4112(%rbp), %rdx
	leaq	-8208(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	cria_vector
	leaq	.LC54(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$37, -8232(%rbp)
	jmp	.L473
.L439:
	leaq	-8288(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$5, -8232(%rbp)
	jmp	.L473
.L466:
	movl	-8288(%rbp), %eax
	cmpl	$1, %eax
	jne	.L497
	movq	$48, -8232(%rbp)
	jmp	.L473
.L497:
	movq	$54, -8232(%rbp)
	jmp	.L473
.L446:
	movl	-8288(%rbp), %eax
	cmpl	$8, %eax
	jne	.L499
	movq	$41, -8232(%rbp)
	jmp	.L473
.L499:
	movq	$10, -8232(%rbp)
	jmp	.L473
.L443:
	addl	$1, -8252(%rbp)
	movq	$12, -8232(%rbp)
	jmp	.L473
.L441:
	leaq	.LC55(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$10, %edi
	call	putchar@PLT
	leaq	.LC56(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$10, %edi
	call	putchar@PLT
	leaq	.LC55(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	call	mil_selection_sort
	leaq	.LC55(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	call	mil_quick_sort_ultimoelemento
	leaq	.LC55(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	call	mil_quick_sort_mediano
	leaq	.LC55(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	call	mil_shell_sort_padrao
	leaq	.LC55(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	call	mil_shell_sort_knuth
	leaq	.LC55(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	call	mil_pesquisa_sequencial
	leaq	.LC55(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	call	mil_pesquisa_binaria
	leaq	.LC55(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$37, -8232(%rbp)
	jmp	.L473
.L461:
	movl	-8288(%rbp), %eax
	cmpl	$9, %eax
	jne	.L501
	movq	$1, -8232(%rbp)
	jmp	.L473
.L501:
	movq	$37, -8232(%rbp)
	jmp	.L473
.L437:
	movl	$2048, %esi
	movl	$0, %edi
	call	aleat
	movq	%rax, -8224(%rbp)
	movq	-8224(%rbp), %rax
	movl	%eax, -8260(%rbp)
	movq	$11, -8232(%rbp)
	jmp	.L473
.L464:
	leaq	-8208(%rbp), %rax
	movl	$1024, %esi
	movq	%rax, %rdi
	call	selection_sort
	movq	count_comparacoes(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC45(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	count_trocas(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC46(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-8208(%rbp), %rax
	movq	%rax, %rdi
	call	imprime_parte
	movq	$0, count_comparacoes(%rip)
	movq	$0, count_trocas(%rip)
	movq	$37, -8232(%rbp)
	jmp	.L473
.L440:
	cmpl	$-1, -8248(%rbp)
	je	.L503
	movq	$26, -8232(%rbp)
	jmp	.L473
.L503:
	movq	$49, -8232(%rbp)
	jmp	.L473
.L469:
	movl	-8268(%rbp), %eax
	cmpl	$1, %eax
	jne	.L505
	movq	$57, -8232(%rbp)
	jmp	.L473
.L505:
	movq	$37, -8232(%rbp)
	jmp	.L473
.L452:
	leaq	.LC50(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-8272(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$3, -8232(%rbp)
	jmp	.L473
.L510:
	nop
.L473:
	jmp	.L507
.L509:
	call	__stack_chk_fail@PLT
.L508:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE24:
	.size	main, .-main
	.globl	desvio_padrao
	.type	desvio_padrao, @function
desvio_padrao:
.LFB25:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$56, %rsp
	movq	%rdi, -56(%rbp)
	movq	$5, -16(%rbp)
.L524:
	cmpq	$9, -16(%rbp)
	ja	.L526
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L514(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L514(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L514:
	.long	.L526-.L514
	.long	.L519-.L514
	.long	.L518-.L514
	.long	.L526-.L514
	.long	.L526-.L514
	.long	.L517-.L514
	.long	.L526-.L514
	.long	.L516-.L514
	.long	.L515-.L514
	.long	.L513-.L514
	.text
.L515:
	movq	-24(%rbp), %rcx
	movabsq	$2361183241434822607, %rdx
	movq	%rcx, %rax
	imulq	%rdx
	movq	%rdx, %rax
	sarq	$7, %rax
	sarq	$63, %rcx
	movq	%rcx, %rdx
	subq	%rdx, %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	%eax, %edi
	call	raiz_quadrada
	movl	%eax, -40(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L520
.L519:
	movl	-40(%rbp), %eax
	jmp	.L525
.L513:
	cmpl	$999, -44(%rbp)
	jg	.L522
	movq	$2, -16(%rbp)
	jmp	.L520
.L522:
	movq	$8, -16(%rbp)
	jmp	.L520
.L517:
	movq	$7, -16(%rbp)
	jmp	.L520
.L516:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	media
	movl	%eax, -36(%rbp)
	movl	-36(%rbp), %eax
	cltq
	movq	%rax, -32(%rbp)
	movq	$0, -24(%rbp)
	movl	$0, -44(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L520
.L518:
	movl	-44(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cltq
	subq	-32(%rbp), %rax
	movq	%rax, %rdx
	movl	-44(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-56(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	cltq
	subq	-32(%rbp), %rax
	imulq	%rdx, %rax
	addq	%rax, -24(%rbp)
	addl	$1, -44(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L520
.L526:
	nop
.L520:
	jmp	.L524
.L525:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE25:
	.size	desvio_padrao, .-desvio_padrao
	.globl	quick_sort
	.type	quick_sort, @function
quick_sort:
.LFB26:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movl	%edx, -32(%rbp)
	movl	%ecx, -36(%rbp)
	movq	$3, -8(%rbp)
.L536:
	cmpq	$3, -8(%rbp)
	je	.L528
	cmpq	$3, -8(%rbp)
	ja	.L537
	cmpq	$0, -8(%rbp)
	je	.L530
	cmpq	$2, -8(%rbp)
	je	.L538
	jmp	.L537
.L528:
	movl	-28(%rbp), %eax
	cmpl	-32(%rbp), %eax
	jge	.L532
	movq	$0, -8(%rbp)
	jmp	.L534
.L532:
	movq	$2, -8(%rbp)
	jmp	.L534
.L530:
	movl	-36(%rbp), %ecx
	movl	-32(%rbp), %edx
	movl	-28(%rbp), %esi
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	particao
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	leal	-1(%rax), %edi
	movl	-36(%rbp), %edx
	movl	-28(%rbp), %esi
	movq	-24(%rbp), %rax
	movl	%edx, %ecx
	movl	%edi, %edx
	movq	%rax, %rdi
	call	quick_sort
	movl	-12(%rbp), %eax
	leal	1(%rax), %esi
	movl	-36(%rbp), %ecx
	movl	-32(%rbp), %edx
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	quick_sort
	movq	$2, -8(%rbp)
	jmp	.L534
.L537:
	nop
.L534:
	jmp	.L536
.L538:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE26:
	.size	quick_sort, .-quick_sort
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
