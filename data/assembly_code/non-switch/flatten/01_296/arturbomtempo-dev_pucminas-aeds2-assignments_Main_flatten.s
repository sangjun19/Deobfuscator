	.file	"arturbomtempo-dev_pucminas-aeds2-assignments_Main_flatten.c"
	.text
	.globl	root
	.bss
	.align 8
	.type	root, @object
	.size	root, 8
root:
	.zero	8
	.globl	_TIG_IZ_33ko_argv
	.align 8
	.type	_TIG_IZ_33ko_argv, @object
	.size	_TIG_IZ_33ko_argv, 8
_TIG_IZ_33ko_argv:
	.zero	8
	.globl	_TIG_IZ_33ko_envp
	.align 8
	.type	_TIG_IZ_33ko_envp, @object
	.size	_TIG_IZ_33ko_envp, 8
_TIG_IZ_33ko_envp:
	.zero	8
	.globl	_TIG_IZ_33ko_argc
	.align 4
	.type	_TIG_IZ_33ko_argc, @object
	.size	_TIG_IZ_33ko_argc, 4
_TIG_IZ_33ko_argc:
	.zero	4
	.text
	.globl	newNode
	.type	newNode, @function
newNode:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movq	$2, -24(%rbp)
.L7:
	cmpq	$2, -24(%rbp)
	je	.L2
	cmpq	$2, -24(%rbp)
	ja	.L9
	cmpq	$0, -24(%rbp)
	je	.L4
	cmpq	$1, -24(%rbp)
	jne	.L9
	movl	$24, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movl	-36(%rbp), %edx
	movl	%edx, (%rax)
	movq	$0, -8(%rbp)
	movq	-32(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	-32(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	$0, -24(%rbp)
	jmp	.L5
.L4:
	movq	-32(%rbp), %rax
	jmp	.L8
.L2:
	movq	$1, -24(%rbp)
	jmp	.L5
.L9:
	nop
.L5:
	jmp	.L7
.L8:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	newNode, .-newNode
	.globl	insert
	.type	insert, @function
insert:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$1, -8(%rbp)
.L15:
	cmpq	$0, -8(%rbp)
	je	.L16
	cmpq	$1, -8(%rbp)
	jne	.L17
	movq	root(%rip), %rdx
	movl	-20(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	recursiveInsert
	movq	%rax, root(%rip)
	movq	$0, -8(%rbp)
	jmp	.L13
.L17:
	nop
.L13:
	jmp	.L15
.L16:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	insert, .-insert
	.section	.rodata
.LC0:
	.string	"%d "
	.text
	.globl	recursivePostOrderWalk
	.type	recursivePostOrderWalk, @function
recursivePostOrderWalk:
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
.L26:
	cmpq	$2, -8(%rbp)
	je	.L19
	cmpq	$2, -8(%rbp)
	ja	.L27
	cmpq	$0, -8(%rbp)
	je	.L28
	cmpq	$1, -8(%rbp)
	jne	.L27
	cmpq	$0, -24(%rbp)
	je	.L22
	movq	$2, -8(%rbp)
	jmp	.L24
.L22:
	movq	$0, -8(%rbp)
	jmp	.L24
.L19:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdi
	call	recursivePostOrderWalk
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, %rdi
	call	recursivePostOrderWalk
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -8(%rbp)
	jmp	.L24
.L27:
	nop
.L24:
	jmp	.L26
.L28:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	recursivePostOrderWalk, .-recursivePostOrderWalk
	.section	.rodata
	.align 8
.LC1:
	.string	"Erro ao remover elemento da \303\241rvore bin\303\241ria."
	.text
	.globl	recursiveRemoval
	.type	recursiveRemoval, @function
recursiveRemoval:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	$13, -24(%rbp)
.L56:
	cmpq	$13, -24(%rbp)
	ja	.L57
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L32(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L32(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L32:
	.long	.L58-.L32
	.long	.L42-.L32
	.long	.L41-.L32
	.long	.L40-.L32
	.long	.L39-.L32
	.long	.L38-.L32
	.long	.L37-.L32
	.long	.L36-.L32
	.long	.L35-.L32
	.long	.L34-.L32
	.long	.L57-.L32
	.long	.L57-.L32
	.long	.L33-.L32
	.long	.L31-.L32
	.text
.L39:
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	leaq	8(%rax), %rdx
	movq	-48(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	biggerLeft
	movq	$0, -24(%rbp)
	jmp	.L44
.L33:
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	leaq	16(%rax), %rdx
	movl	-36(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	recursiveRemoval
	movq	$0, -24(%rbp)
	jmp	.L44
.L35:
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	movq	16(%rax), %rdx
	movq	-48(%rbp), %rax
	movq	%rdx, (%rax)
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$0, -24(%rbp)
	jmp	.L44
.L42:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rsi
	movl	$1, %edi
	movl	$0, %eax
	call	errx@PLT
.L40:
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	movl	(%rax), %eax
	cmpl	%eax, -36(%rbp)
	jge	.L45
	movq	$6, -24(%rbp)
	jmp	.L44
.L45:
	movq	$9, -24(%rbp)
	jmp	.L44
.L34:
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	movl	(%rax), %eax
	cmpl	%eax, -36(%rbp)
	jle	.L47
	movq	$12, -24(%rbp)
	jmp	.L44
.L47:
	movq	$5, -24(%rbp)
	jmp	.L44
.L31:
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L49
	movq	$1, -24(%rbp)
	jmp	.L44
.L49:
	movq	$3, -24(%rbp)
	jmp	.L44
.L37:
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	leaq	8(%rax), %rdx
	movl	-36(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	recursiveRemoval
	movq	$0, -24(%rbp)
	jmp	.L44
.L38:
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	movq	16(%rax), %rax
	testq	%rax, %rax
	jne	.L51
	movq	$2, -24(%rbp)
	jmp	.L44
.L51:
	movq	$7, -24(%rbp)
	jmp	.L44
.L36:
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	jne	.L54
	movq	$8, -24(%rbp)
	jmp	.L44
.L54:
	movq	$4, -24(%rbp)
	jmp	.L44
.L41:
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -8(%rbp)
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rdx
	movq	-48(%rbp), %rax
	movq	%rdx, (%rax)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$0, -24(%rbp)
	jmp	.L44
.L57:
	nop
.L44:
	jmp	.L56
.L58:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	recursiveRemoval, .-recursiveRemoval
	.globl	recursiveCentralWalk
	.type	recursiveCentralWalk, @function
recursiveCentralWalk:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$3, -8(%rbp)
.L68:
	cmpq	$3, -8(%rbp)
	je	.L60
	cmpq	$3, -8(%rbp)
	ja	.L69
	cmpq	$1, -8(%rbp)
	je	.L62
	cmpq	$2, -8(%rbp)
	je	.L70
	jmp	.L69
.L62:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdi
	call	recursiveCentralWalk
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, %rdi
	call	recursiveCentralWalk
	movq	$2, -8(%rbp)
	jmp	.L64
.L60:
	cmpq	$0, -24(%rbp)
	je	.L65
	movq	$1, -8(%rbp)
	jmp	.L64
.L65:
	movq	$2, -8(%rbp)
	jmp	.L64
.L69:
	nop
.L64:
	jmp	.L68
.L70:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	recursiveCentralWalk, .-recursiveCentralWalk
	.globl	removal
	.type	removal, @function
removal:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$1, -8(%rbp)
.L76:
	cmpq	$0, -8(%rbp)
	je	.L77
	cmpq	$1, -8(%rbp)
	jne	.L78
	movl	-20(%rbp), %eax
	leaq	root(%rip), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	recursiveRemoval
	movq	$0, -8(%rbp)
	jmp	.L74
.L78:
	nop
.L74:
	jmp	.L76
.L77:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	removal, .-removal
	.section	.rodata
	.align 8
.LC2:
	.string	"Erro ao inserir elemento na \303\241rvore bin\303\241ria"
	.text
	.globl	recursiveInsert
	.type	recursiveInsert, @function
recursiveInsert:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$1, -8(%rbp)
.L98:
	cmpq	$7, -8(%rbp)
	ja	.L100
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L82(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L82(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L82:
	.long	.L89-.L82
	.long	.L88-.L82
	.long	.L87-.L82
	.long	.L86-.L82
	.long	.L85-.L82
	.long	.L84-.L82
	.long	.L83-.L82
	.long	.L81-.L82
	.text
.L85:
	movq	-32(%rbp), %rax
	jmp	.L99
.L88:
	cmpq	$0, -32(%rbp)
	jne	.L91
	movq	$7, -8(%rbp)
	jmp	.L93
.L91:
	movq	$2, -8(%rbp)
	jmp	.L93
.L86:
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -20(%rbp)
	jle	.L94
	movq	$6, -8(%rbp)
	jmp	.L93
.L94:
	movq	$0, -8(%rbp)
	jmp	.L93
.L83:
	movq	-32(%rbp), %rax
	movq	16(%rax), %rdx
	movl	-20(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	recursiveInsert
	movq	-32(%rbp), %rdx
	movq	%rax, 16(%rdx)
	movq	$4, -8(%rbp)
	jmp	.L93
.L84:
	movq	-32(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-20(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	recursiveInsert
	movq	-32(%rbp), %rdx
	movq	%rax, 8(%rdx)
	movq	$4, -8(%rbp)
	jmp	.L93
.L89:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rsi
	movl	$1, %edi
	movl	$0, %eax
	call	errx@PLT
.L81:
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	newNode
	movq	%rax, -32(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L93
.L87:
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -20(%rbp)
	jge	.L96
	movq	$5, -8(%rbp)
	jmp	.L93
.L96:
	movq	$3, -8(%rbp)
	jmp	.L93
.L100:
	nop
.L93:
	jmp	.L98
.L99:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	recursiveInsert, .-recursiveInsert
	.globl	start
	.type	start, @function
start:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L106:
	cmpq	$0, -8(%rbp)
	je	.L102
	cmpq	$1, -8(%rbp)
	jne	.L108
	jmp	.L107
.L102:
	movq	$0, root(%rip)
	movq	$1, -8(%rbp)
	jmp	.L105
.L108:
	nop
.L105:
	jmp	.L106
.L107:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	start, .-start
	.section	.rodata
.LC3:
	.string	"[ "
.LC4:
	.string	"]"
	.text
	.globl	preOrderWalk
	.type	preOrderWalk, @function
preOrderWalk:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L115:
	cmpq	$2, -8(%rbp)
	je	.L110
	cmpq	$2, -8(%rbp)
	ja	.L116
	cmpq	$0, -8(%rbp)
	je	.L117
	cmpq	$1, -8(%rbp)
	jne	.L116
	movq	$2, -8(%rbp)
	jmp	.L113
.L110:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	root(%rip), %rax
	movq	%rax, %rdi
	call	recursivePreOrderWalk
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L113
.L116:
	nop
.L113:
	jmp	.L115
.L117:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	preOrderWalk, .-preOrderWalk
	.globl	centralWalk
	.type	centralWalk, @function
centralWalk:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L124:
	cmpq	$2, -8(%rbp)
	je	.L125
	cmpq	$2, -8(%rbp)
	ja	.L126
	cmpq	$0, -8(%rbp)
	je	.L121
	cmpq	$1, -8(%rbp)
	jne	.L126
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	root(%rip), %rax
	movq	%rax, %rdi
	call	recursiveCentralWalk
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -8(%rbp)
	jmp	.L122
.L121:
	movq	$1, -8(%rbp)
	jmp	.L122
.L126:
	nop
.L122:
	jmp	.L124
.L125:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	centralWalk, .-centralWalk
	.globl	search
	.type	search, @function
search:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$0, -8(%rbp)
.L132:
	cmpq	$0, -8(%rbp)
	je	.L128
	cmpq	$1, -8(%rbp)
	jne	.L134
	movzbl	-9(%rbp), %eax
	jmp	.L133
.L128:
	movq	root(%rip), %rdx
	movl	-20(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	recursiveSearch
	movb	%al, -9(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L131
.L134:
	nop
.L131:
	jmp	.L132
.L133:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	search, .-search
	.section	.rodata
	.align 8
.LC5:
	.string	"\nInserir: 3, 5, 1, 8, 2, 4, 7 e 6"
.LC6:
	.string	"\nCaminhar: central, pre e pos"
.LC7:
	.string	"\nRemover: 2"
	.align 8
.LC8:
	.string	"\nVoltando com a \303\241rvore inicial"
.LC9:
	.string	"\nRemover: 1"
.LC10:
	.string	"\nRemover: 3"
	.text
	.globl	main
	.type	main, @function
main:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, root(%rip)
	nop
.L136:
	movq	$0, _TIG_IZ_33ko_envp(%rip)
	nop
.L137:
	movq	$0, _TIG_IZ_33ko_argv(%rip)
	nop
.L138:
	movl	$0, _TIG_IZ_33ko_argc(%rip)
	nop
	nop
.L139:
.L140:
#APP
# 118 "arturbomtempo-dev_pucminas-aeds2-assignments_Main.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-33ko--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_33ko_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_33ko_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_33ko_envp(%rip)
	nop
	movq	$0, -8(%rbp)
.L146:
	cmpq	$2, -8(%rbp)
	je	.L141
	cmpq	$2, -8(%rbp)
	ja	.L148
	cmpq	$0, -8(%rbp)
	je	.L143
	cmpq	$1, -8(%rbp)
	jne	.L148
	movl	$0, %eax
	jmp	.L147
.L143:
	movq	$2, -8(%rbp)
	jmp	.L145
.L141:
	call	start
	movl	$3, %edi
	call	insert
	movl	$5, %edi
	call	insert
	movl	$1, %edi
	call	insert
	movl	$8, %edi
	call	insert
	movl	$2, %edi
	call	insert
	movl	$4, %edi
	call	insert
	movl	$7, %edi
	call	insert
	movl	$6, %edi
	call	insert
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	call	centralWalk
	call	preOrderWalk
	call	postOrderWalk
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$2, %edi
	call	removal
	call	centralWalk
	call	preOrderWalk
	call	postOrderWalk
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$2, %edi
	call	insert
	movl	$1, %edi
	call	removal
	call	centralWalk
	call	preOrderWalk
	call	postOrderWalk
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$2, %edi
	call	removal
	movl	$1, %edi
	call	insert
	movl	$2, %edi
	call	insert
	movl	$3, %edi
	call	removal
	call	centralWalk
	call	preOrderWalk
	call	postOrderWalk
	movq	$1, -8(%rbp)
	jmp	.L145
.L148:
	nop
.L145:
	jmp	.L146
.L147:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	main, .-main
	.globl	postOrderWalk
	.type	postOrderWalk, @function
postOrderWalk:
.LFB17:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L155:
	cmpq	$2, -8(%rbp)
	je	.L156
	cmpq	$2, -8(%rbp)
	ja	.L157
	cmpq	$0, -8(%rbp)
	je	.L152
	cmpq	$1, -8(%rbp)
	jne	.L157
	movq	$0, -8(%rbp)
	jmp	.L153
.L152:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	root(%rip), %rax
	movq	%rax, %rdi
	call	recursivePostOrderWalk
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -8(%rbp)
	jmp	.L153
.L157:
	nop
.L153:
	jmp	.L155
.L156:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE17:
	.size	postOrderWalk, .-postOrderWalk
	.globl	recursivePreOrderWalk
	.type	recursivePreOrderWalk, @function
recursivePreOrderWalk:
.LFB18:
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
.L166:
	cmpq	$2, -8(%rbp)
	je	.L159
	cmpq	$2, -8(%rbp)
	ja	.L167
	cmpq	$0, -8(%rbp)
	je	.L168
	cmpq	$1, -8(%rbp)
	jne	.L167
	cmpq	$0, -24(%rbp)
	je	.L162
	movq	$2, -8(%rbp)
	jmp	.L164
.L162:
	movq	$0, -8(%rbp)
	jmp	.L164
.L159:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdi
	call	recursivePreOrderWalk
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, %rdi
	call	recursivePreOrderWalk
	movq	$0, -8(%rbp)
	jmp	.L164
.L167:
	nop
.L164:
	jmp	.L166
.L168:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE18:
	.size	recursivePreOrderWalk, .-recursivePreOrderWalk
	.globl	biggerLeft
	.type	biggerLeft, @function
biggerLeft:
.LFB19:
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
	movq	$0, -16(%rbp)
.L179:
	cmpq	$4, -16(%rbp)
	je	.L170
	cmpq	$4, -16(%rbp)
	ja	.L180
	cmpq	$2, -16(%rbp)
	je	.L172
	cmpq	$2, -16(%rbp)
	ja	.L180
	cmpq	$0, -16(%rbp)
	je	.L173
	cmpq	$1, -16(%rbp)
	je	.L181
	jmp	.L180
.L170:
	movq	-32(%rbp), %rax
	movq	(%rax), %rax
	leaq	16(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	biggerLeft
	movq	$1, -16(%rbp)
	jmp	.L175
.L173:
	movq	-32(%rbp), %rax
	movq	(%rax), %rax
	movq	16(%rax), %rax
	testq	%rax, %rax
	je	.L177
	movq	$4, -16(%rbp)
	jmp	.L175
.L177:
	movq	$2, -16(%rbp)
	jmp	.L175
.L172:
	movq	-32(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -8(%rbp)
	movq	-32(%rbp), %rax
	movq	(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movl	(%rdx), %edx
	movl	%edx, (%rax)
	movq	-32(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, (%rax)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$1, -16(%rbp)
	jmp	.L175
.L180:
	nop
.L175:
	jmp	.L179
.L181:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE19:
	.size	biggerLeft, .-biggerLeft
	.globl	recursiveSearch
	.type	recursiveSearch, @function
recursiveSearch:
.LFB21:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$8, -8(%rbp)
.L204:
	cmpq	$8, -8(%rbp)
	ja	.L206
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L185(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L185(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L185:
	.long	.L193-.L185
	.long	.L192-.L185
	.long	.L191-.L185
	.long	.L190-.L185
	.long	.L189-.L185
	.long	.L188-.L185
	.long	.L187-.L185
	.long	.L186-.L185
	.long	.L184-.L185
	.text
.L189:
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -20(%rbp)
	jne	.L194
	movq	$3, -8(%rbp)
	jmp	.L196
.L194:
	movq	$0, -8(%rbp)
	jmp	.L196
.L184:
	cmpq	$0, -32(%rbp)
	jne	.L197
	movq	$2, -8(%rbp)
	jmp	.L196
.L197:
	movq	$4, -8(%rbp)
	jmp	.L196
.L192:
	movzbl	-9(%rbp), %eax
	jmp	.L205
.L190:
	movb	$1, -9(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L196
.L187:
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -20(%rbp)
	jle	.L200
	movq	$7, -8(%rbp)
	jmp	.L196
.L200:
	movq	$1, -8(%rbp)
	jmp	.L196
.L188:
	movq	-32(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-20(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	recursiveSearch
	movb	%al, -9(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L196
.L193:
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -20(%rbp)
	jge	.L202
	movq	$5, -8(%rbp)
	jmp	.L196
.L202:
	movq	$6, -8(%rbp)
	jmp	.L196
.L186:
	movq	-32(%rbp), %rax
	movq	16(%rax), %rdx
	movl	-20(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	recursiveSearch
	movb	%al, -9(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L196
.L191:
	movb	$0, -9(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L196
.L206:
	nop
.L196:
	jmp	.L204
.L205:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE21:
	.size	recursiveSearch, .-recursiveSearch
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
