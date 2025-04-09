	.file	"SkyRats_IMAV2022-NanocopterAi_uc_flatten.c"
	.text
	.globl	_TIG_IZ_mo9M_argv
	.bss
	.align 8
	.type	_TIG_IZ_mo9M_argv, @object
	.size	_TIG_IZ_mo9M_argv, 8
_TIG_IZ_mo9M_argv:
	.zero	8
	.globl	_TIG_IZ_mo9M_argc
	.align 4
	.type	_TIG_IZ_mo9M_argc, @object
	.size	_TIG_IZ_mo9M_argc, 4
_TIG_IZ_mo9M_argc:
	.zero	4
	.globl	cost_matrix
	.align 32
	.type	cost_matrix, @object
	.size	cost_matrix, 400
cost_matrix:
	.zero	400
	.globl	_TIG_IZ_mo9M_envp
	.align 8
	.type	_TIG_IZ_mo9M_envp, @object
	.size	_TIG_IZ_mo9M_envp, 8
_TIG_IZ_mo9M_envp:
	.zero	8
	.text
	.globl	is_empty
	.type	is_empty, @function
is_empty:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$2, -8(%rbp)
.L10:
	cmpq	$3, -8(%rbp)
	je	.L2
	cmpq	$3, -8(%rbp)
	ja	.L12
	cmpq	$2, -8(%rbp)
	je	.L4
	cmpq	$2, -8(%rbp)
	ja	.L12
	cmpq	$0, -8(%rbp)
	je	.L5
	cmpq	$1, -8(%rbp)
	jne	.L12
	movl	$1, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L6
.L2:
	movl	-12(%rbp), %eax
	jmp	.L11
.L5:
	cmpq	$0, -24(%rbp)
	jne	.L8
	movq	$1, -8(%rbp)
	jmp	.L6
.L8:
	movq	$3, -8(%rbp)
	jmp	.L6
.L4:
	movl	$0, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L6
.L12:
	nop
.L6:
	jmp	.L10
.L11:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	is_empty, .-is_empty
	.section	.rodata
.LC0:
	.string	"\n\nFreeing nodes in a list"
	.text
	.globl	free_nodes_in_list
	.type	free_nodes_in_list, @function
free_nodes_in_list:
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
	movq	$4, -16(%rbp)
.L23:
	cmpq	$5, -16(%rbp)
	je	.L14
	cmpq	$5, -16(%rbp)
	ja	.L24
	cmpq	$4, -16(%rbp)
	je	.L16
	cmpq	$4, -16(%rbp)
	ja	.L24
	cmpq	$1, -16(%rbp)
	je	.L17
	cmpq	$3, -16(%rbp)
	je	.L25
	jmp	.L24
.L16:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -16(%rbp)
	jmp	.L19
.L17:
	movq	-24(%rbp), %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	print_node
	movq	-8(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$5, -16(%rbp)
	jmp	.L19
.L14:
	cmpq	$0, -24(%rbp)
	je	.L21
	movq	$1, -16(%rbp)
	jmp	.L19
.L21:
	movq	$3, -16(%rbp)
	jmp	.L19
.L24:
	nop
.L19:
	jmp	.L23
.L25:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	free_nodes_in_list, .-free_nodes_in_list
	.section	.rodata
.LC1:
	.string	"\nusage: uc goal-node"
	.text
	.globl	print_usage
	.type	print_usage, @function
print_usage:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L31:
	cmpq	$0, -8(%rbp)
	je	.L27
	cmpq	$1, -8(%rbp)
	jne	.L33
	jmp	.L32
.L27:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L30
.L33:
	nop
.L30:
	jmp	.L31
.L32:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	print_usage, .-print_usage
	.section	.rodata
	.align 8
.LC2:
	.string	"\nThe OPEN list is empty, no solution exists"
	.text
	.globl	no_solution_exists
	.type	no_solution_exists, @function
no_solution_exists:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$2, -8(%rbp)
.L40:
	cmpq	$2, -8(%rbp)
	je	.L35
	cmpq	$2, -8(%rbp)
	ja	.L42
	cmpq	$0, -8(%rbp)
	je	.L37
	cmpq	$1, -8(%rbp)
	jne	.L42
	movq	-24(%rbp), %rax
	movl	$0, (%rax)
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -8(%rbp)
	jmp	.L38
.L37:
	movl	$1, %eax
	jmp	.L41
.L35:
	movq	$1, -8(%rbp)
	jmp	.L38
.L42:
	nop
.L38:
	jmp	.L40
.L41:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	no_solution_exists, .-no_solution_exists
	.section	.rodata
.LC3:
	.string	"\n"
.LC4:
	.string	"\n\nTraversing list"
	.text
	.globl	traverse_list
	.type	traverse_list, @function
traverse_list:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$2, -16(%rbp)
.L55:
	cmpq	$5, -16(%rbp)
	ja	.L56
	movq	-16(%rbp), %rax
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
	.long	.L56-.L46
	.long	.L50-.L46
	.long	.L49-.L46
	.long	.L48-.L46
	.long	.L47-.L46
	.long	.L57-.L46
	.text
.L47:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -16(%rbp)
	jmp	.L51
.L50:
	movq	-24(%rbp), %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	print_node
	movq	-8(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L51
.L48:
	cmpq	$0, -24(%rbp)
	je	.L52
	movq	$1, -16(%rbp)
	jmp	.L51
.L52:
	movq	$4, -16(%rbp)
	jmp	.L51
.L49:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L51
.L56:
	nop
.L51:
	jmp	.L55
.L57:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	traverse_list, .-traverse_list
	.globl	main
	.type	main, @function
main:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$-1, cost_matrix(%rip)
	movl	$5, 4+cost_matrix(%rip)
	movl	$5, 8+cost_matrix(%rip)
	movl	$6, 12+cost_matrix(%rip)
	movl	$-1, 16+cost_matrix(%rip)
	movl	$7, 20+cost_matrix(%rip)
	movl	$-1, 24+cost_matrix(%rip)
	movl	$-1, 28+cost_matrix(%rip)
	movl	$-1, 32+cost_matrix(%rip)
	movl	$-1, 36+cost_matrix(%rip)
	movl	$5, 40+cost_matrix(%rip)
	movl	$-1, 44+cost_matrix(%rip)
	movl	$-1, 48+cost_matrix(%rip)
	movl	$-1, 52+cost_matrix(%rip)
	movl	$-1, 56+cost_matrix(%rip)
	movl	$-1, 60+cost_matrix(%rip)
	movl	$-1, 64+cost_matrix(%rip)
	movl	$4, 68+cost_matrix(%rip)
	movl	$-1, 72+cost_matrix(%rip)
	movl	$-1, 76+cost_matrix(%rip)
	movl	$5, 80+cost_matrix(%rip)
	movl	$-1, 84+cost_matrix(%rip)
	movl	$-1, 88+cost_matrix(%rip)
	movl	$-1, 92+cost_matrix(%rip)
	movl	$6, 96+cost_matrix(%rip)
	movl	$-1, 100+cost_matrix(%rip)
	movl	$-1, 104+cost_matrix(%rip)
	movl	$-1, 108+cost_matrix(%rip)
	movl	$-1, 112+cost_matrix(%rip)
	movl	$-1, 116+cost_matrix(%rip)
	movl	$6, 120+cost_matrix(%rip)
	movl	$-1, 124+cost_matrix(%rip)
	movl	$-1, 128+cost_matrix(%rip)
	movl	$-1, 132+cost_matrix(%rip)
	movl	$5, 136+cost_matrix(%rip)
	movl	$-1, 140+cost_matrix(%rip)
	movl	$-1, 144+cost_matrix(%rip)
	movl	$-1, 148+cost_matrix(%rip)
	movl	$2, 152+cost_matrix(%rip)
	movl	$-1, 156+cost_matrix(%rip)
	movl	$-1, 160+cost_matrix(%rip)
	movl	$-1, 164+cost_matrix(%rip)
	movl	$6, 168+cost_matrix(%rip)
	movl	$5, 172+cost_matrix(%rip)
	movl	$-1, 176+cost_matrix(%rip)
	movl	$-1, 180+cost_matrix(%rip)
	movl	$4, 184+cost_matrix(%rip)
	movl	$-1, 188+cost_matrix(%rip)
	movl	$-1, 192+cost_matrix(%rip)
	movl	$5, 196+cost_matrix(%rip)
	movl	$7, 200+cost_matrix(%rip)
	movl	$-1, 204+cost_matrix(%rip)
	movl	$-1, 208+cost_matrix(%rip)
	movl	$-1, 212+cost_matrix(%rip)
	movl	$-1, 216+cost_matrix(%rip)
	movl	$-1, 220+cost_matrix(%rip)
	movl	$3, 224+cost_matrix(%rip)
	movl	$-1, 228+cost_matrix(%rip)
	movl	$-1, 232+cost_matrix(%rip)
	movl	$-1, 236+cost_matrix(%rip)
	movl	$-1, 240+cost_matrix(%rip)
	movl	$-1, 244+cost_matrix(%rip)
	movl	$-1, 248+cost_matrix(%rip)
	movl	$-1, 252+cost_matrix(%rip)
	movl	$4, 256+cost_matrix(%rip)
	movl	$3, 260+cost_matrix(%rip)
	movl	$-1, 264+cost_matrix(%rip)
	movl	$-1, 268+cost_matrix(%rip)
	movl	$-1, 272+cost_matrix(%rip)
	movl	$-1, 276+cost_matrix(%rip)
	movl	$-1, 280+cost_matrix(%rip)
	movl	$4, 284+cost_matrix(%rip)
	movl	$-1, 288+cost_matrix(%rip)
	movl	$-1, 292+cost_matrix(%rip)
	movl	$-1, 296+cost_matrix(%rip)
	movl	$-1, 300+cost_matrix(%rip)
	movl	$-1, 304+cost_matrix(%rip)
	movl	$-1, 308+cost_matrix(%rip)
	movl	$-1, 312+cost_matrix(%rip)
	movl	$2, 316+cost_matrix(%rip)
	movl	$-1, 320+cost_matrix(%rip)
	movl	$-1, 324+cost_matrix(%rip)
	movl	$-1, 328+cost_matrix(%rip)
	movl	$2, 332+cost_matrix(%rip)
	movl	$-1, 336+cost_matrix(%rip)
	movl	$-1, 340+cost_matrix(%rip)
	movl	$-1, 344+cost_matrix(%rip)
	movl	$-1, 348+cost_matrix(%rip)
	movl	$-1, 352+cost_matrix(%rip)
	movl	$1, 356+cost_matrix(%rip)
	movl	$-1, 360+cost_matrix(%rip)
	movl	$-1, 364+cost_matrix(%rip)
	movl	$-1, 368+cost_matrix(%rip)
	movl	$-1, 372+cost_matrix(%rip)
	movl	$5, 376+cost_matrix(%rip)
	movl	$-1, 380+cost_matrix(%rip)
	movl	$-1, 384+cost_matrix(%rip)
	movl	$2, 388+cost_matrix(%rip)
	movl	$1, 392+cost_matrix(%rip)
	movl	$-1, 396+cost_matrix(%rip)
	nop
.L59:
	movq	$0, _TIG_IZ_mo9M_envp(%rip)
	nop
.L60:
	movq	$0, _TIG_IZ_mo9M_argv(%rip)
	nop
.L61:
	movl	$0, _TIG_IZ_mo9M_argc(%rip)
	nop
	nop
.L62:
.L63:
#APP
# 637 "SkyRats_IMAV2022-NanocopterAi_uc.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-mo9M--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_mo9M_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_mo9M_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_mo9M_envp(%rip)
	nop
	movq	$32, -16(%rbp)
.L107:
	cmpq	$32, -16(%rbp)
	ja	.L110
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L66(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L66(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L66:
	.long	.L88-.L66
	.long	.L87-.L66
	.long	.L86-.L66
	.long	.L110-.L66
	.long	.L110-.L66
	.long	.L110-.L66
	.long	.L85-.L66
	.long	.L84-.L66
	.long	.L83-.L66
	.long	.L110-.L66
	.long	.L110-.L66
	.long	.L82-.L66
	.long	.L81-.L66
	.long	.L110-.L66
	.long	.L80-.L66
	.long	.L79-.L66
	.long	.L78-.L66
	.long	.L77-.L66
	.long	.L76-.L66
	.long	.L75-.L66
	.long	.L110-.L66
	.long	.L110-.L66
	.long	.L74-.L66
	.long	.L73-.L66
	.long	.L110-.L66
	.long	.L72-.L66
	.long	.L110-.L66
	.long	.L71-.L66
	.long	.L70-.L66
	.long	.L69-.L66
	.long	.L68-.L66
	.long	.L67-.L66
	.long	.L65-.L66
	.text
.L76:
	cmpl	$-1, -72(%rbp)
	je	.L89
	movq	$15, -16(%rbp)
	jmp	.L91
.L89:
	movq	$1, -16(%rbp)
	jmp	.L91
.L72:
	movl	-76(%rbp), %eax
	testl	%eax, %eax
	je	.L92
	movq	$29, -16(%rbp)
	jmp	.L91
.L92:
	movq	$7, -16(%rbp)
	jmp	.L91
.L68:
	movq	-40(%rbp), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	cmpl	%eax, -64(%rbp)
	je	.L94
	movq	$8, -16(%rbp)
	jmp	.L91
.L94:
	movq	$1, -16(%rbp)
	jmp	.L91
.L80:
	movl	$0, -64(%rbp)
	movq	$31, -16(%rbp)
	jmp	.L91
.L79:
	movq	-40(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	jne	.L96
	movq	$28, -16(%rbp)
	jmp	.L91
.L96:
	movq	$30, -16(%rbp)
	jmp	.L91
.L67:
	cmpl	$9, -64(%rbp)
	jg	.L98
	movq	$12, -16(%rbp)
	jmp	.L91
.L98:
	movq	$25, -16(%rbp)
	jmp	.L91
.L81:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	movl	-64(%rbp), %edx
	movslq	%edx, %rcx
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rcx, %rax
	leaq	0(,%rax,4), %rdx
	leaq	cost_matrix(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, -72(%rbp)
	movq	$18, -16(%rbp)
	jmp	.L91
.L83:
	movq	-40(%rbp), %rax
	movl	4(%rax), %edx
	movl	-72(%rbp), %eax
	addl	%eax, %edx
	movl	-64(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	create_new_node
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, 8(%rax)
	leaq	-32(%rbp), %rdx
	leaq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	add_to_list
	movq	$1, -16(%rbp)
	jmp	.L91
.L87:
	addl	$1, -64(%rbp)
	movq	$31, -16(%rbp)
	jmp	.L91
.L73:
	leaq	-76(%rbp), %rax
	movq	%rax, %rdi
	call	no_solution_exists
	movq	$25, -16(%rbp)
	jmp	.L91
.L78:
	call	print_usage
	movl	$0, %edi
	call	exit@PLT
.L82:
	movl	$99, -52(%rbp)
	movl	$1, -76(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L91
.L75:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L108
	jmp	.L109
.L65:
	movq	$11, -16(%rbp)
	jmp	.L91
.L77:
	movl	-68(%rbp), %edx
	leaq	-24(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	remove_from_list
	movq	%rax, -40(%rbp)
	leaq	-40(%rbp), %rdx
	leaq	-48(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	add_to_list
	movq	-40(%rbp), %rax
	movl	-68(%rbp), %edx
	movl	%edx, %esi
	movq	%rax, %rdi
	call	is_a_goal_node
	movl	%eax, -60(%rbp)
	movq	$27, -16(%rbp)
	jmp	.L91
.L85:
	movq	-40(%rbp), %rax
	leaq	-76(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	solution_found
	movq	$25, -16(%rbp)
	jmp	.L91
.L71:
	cmpl	$0, -60(%rbp)
	je	.L101
	movq	$6, -16(%rbp)
	jmp	.L91
.L101:
	movq	$14, -16(%rbp)
	jmp	.L91
.L74:
	movq	-96(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -68(%rbp)
	movq	$0, -24(%rbp)
	movq	$0, -48(%rbp)
	movl	$0, %esi
	movl	$0, %edi
	call	create_new_node
	movq	%rax, -40(%rbp)
	leaq	-40(%rbp), %rdx
	leaq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	add_to_list
	movq	$25, -16(%rbp)
	jmp	.L91
.L70:
	movq	-40(%rbp), %rax
	movl	4(%rax), %edx
	movl	-72(%rbp), %eax
	addl	%eax, %edx
	movl	-64(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	create_new_node
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, 8(%rax)
	leaq	-32(%rbp), %rdx
	leaq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	add_to_list
	movq	$1, -16(%rbp)
	jmp	.L91
.L88:
	cmpl	$0, -56(%rbp)
	je	.L103
	movq	$23, -16(%rbp)
	jmp	.L91
.L103:
	movq	$17, -16(%rbp)
	jmp	.L91
.L84:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	free_nodes_in_list
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	free_nodes_in_list
	movq	$19, -16(%rbp)
	jmp	.L91
.L69:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	is_empty
	movl	%eax, -56(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L91
.L86:
	cmpl	$1, -84(%rbp)
	jg	.L105
	movq	$16, -16(%rbp)
	jmp	.L91
.L105:
	movq	$22, -16(%rbp)
	jmp	.L91
.L110:
	nop
.L91:
	jmp	.L107
.L109:
	call	__stack_chk_fail@PLT
.L108:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	main, .-main
	.globl	is_a_goal_node
	.type	is_a_goal_node, @function
is_a_goal_node:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$1, -8(%rbp)
.L120:
	cmpq	$3, -8(%rbp)
	je	.L112
	cmpq	$3, -8(%rbp)
	ja	.L122
	cmpq	$2, -8(%rbp)
	je	.L114
	cmpq	$2, -8(%rbp)
	ja	.L122
	cmpq	$0, -8(%rbp)
	je	.L115
	cmpq	$1, -8(%rbp)
	jne	.L122
	movl	$0, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L116
.L112:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -28(%rbp)
	jne	.L117
	movq	$0, -8(%rbp)
	jmp	.L116
.L117:
	movq	$2, -8(%rbp)
	jmp	.L116
.L115:
	movl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L116
.L114:
	movl	-12(%rbp), %eax
	jmp	.L121
.L122:
	nop
.L116:
	jmp	.L120
.L121:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	is_a_goal_node, .-is_a_goal_node
	.section	.rodata
.LC5:
	.string	"\n\n\nHere is the solution"
	.text
	.globl	solution_found
	.type	solution_found, @function
solution_found:
.LFB12:
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
	movq	$4, -8(%rbp)
.L135:
	cmpq	$6, -8(%rbp)
	ja	.L137
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L126(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L126(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L126:
	.long	.L130-.L126
	.long	.L129-.L126
	.long	.L128-.L126
	.long	.L137-.L126
	.long	.L127-.L126
	.long	.L137-.L126
	.long	.L125-.L126
	.text
.L127:
	movq	$0, -8(%rbp)
	jmp	.L131
.L129:
	cmpq	$0, -16(%rbp)
	je	.L132
	movq	$2, -8(%rbp)
	jmp	.L131
.L132:
	movq	$6, -8(%rbp)
	jmp	.L131
.L125:
	movl	$1, %eax
	jmp	.L136
.L130:
	movq	-24(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	-32(%rbp), %rax
	movl	$0, (%rax)
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L131
.L128:
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	print_node
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L131
.L137:
	nop
.L131:
	jmp	.L135
.L136:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	solution_found, .-solution_found
	.globl	create_new_node
	.type	create_new_node, @function
create_new_node:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movl	%esi, -40(%rbp)
	movq	$2, -16(%rbp)
.L144:
	cmpq	$2, -16(%rbp)
	je	.L139
	cmpq	$2, -16(%rbp)
	ja	.L146
	cmpq	$0, -16(%rbp)
	je	.L141
	cmpq	$1, -16(%rbp)
	jne	.L146
	movl	$24, %esi
	movl	$1, %edi
	call	calloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	-36(%rbp), %edx
	movl	%edx, (%rax)
	movq	-24(%rbp), %rax
	movl	-40(%rbp), %edx
	movl	%edx, 4(%rax)
	movq	-24(%rbp), %rax
	movq	$0, 16(%rax)
	movq	-24(%rbp), %rax
	movq	$0, 8(%rax)
	movq	$0, -16(%rbp)
	jmp	.L142
.L141:
	movq	-24(%rbp), %rax
	jmp	.L145
.L139:
	movq	$1, -16(%rbp)
	jmp	.L142
.L146:
	nop
.L142:
	jmp	.L144
.L145:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	create_new_node, .-create_new_node
	.section	.rodata
.LC6:
	.string	"\nNode number=%d cost=%d"
	.text
	.globl	print_node
	.type	print_node, @function
print_node:
.LFB15:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$0, -8(%rbp)
.L152:
	cmpq	$0, -8(%rbp)
	je	.L148
	cmpq	$1, -8(%rbp)
	jne	.L154
	jmp	.L153
.L148:
	movq	-24(%rbp), %rax
	movl	4(%rax), %edx
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L151
.L154:
	nop
.L151:
	jmp	.L152
.L153:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE15:
	.size	print_node, .-print_node
	.section	.rodata
.LC7:
	.string	"\nEMPTY LIST CANNOT REMOVE "
	.text
	.globl	remove_from_list
	.type	remove_from_list, @function
remove_from_list:
.LFB16:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movl	%esi, -60(%rbp)
	movq	$22, -8(%rbp)
.L209:
	cmpq	$36, -8(%rbp)
	ja	.L211
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L158(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L158(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L158:
	.long	.L211-.L158
	.long	.L184-.L158
	.long	.L183-.L158
	.long	.L182-.L158
	.long	.L211-.L158
	.long	.L181-.L158
	.long	.L180-.L158
	.long	.L179-.L158
	.long	.L178-.L158
	.long	.L211-.L158
	.long	.L211-.L158
	.long	.L177-.L158
	.long	.L176-.L158
	.long	.L175-.L158
	.long	.L174-.L158
	.long	.L211-.L158
	.long	.L173-.L158
	.long	.L172-.L158
	.long	.L211-.L158
	.long	.L211-.L158
	.long	.L171-.L158
	.long	.L170-.L158
	.long	.L169-.L158
	.long	.L168-.L158
	.long	.L167-.L158
	.long	.L166-.L158
	.long	.L211-.L158
	.long	.L211-.L158
	.long	.L165-.L158
	.long	.L164-.L158
	.long	.L163-.L158
	.long	.L162-.L158
	.long	.L161-.L158
	.long	.L160-.L158
	.long	.L159-.L158
	.long	.L211-.L158
	.long	.L157-.L158
	.text
.L166:
	movq	-24(%rbp), %rax
	movl	4(%rax), %eax
	cmpl	%eax, -36(%rbp)
	jle	.L185
	movq	$23, -8(%rbp)
	jmp	.L187
.L185:
	movq	$12, -8(%rbp)
	jmp	.L187
.L163:
	movl	$0, -44(%rbp)
	movl	$0, -40(%rbp)
	movl	$16000, -36(%rbp)
	movl	$0, -28(%rbp)
	movq	$36, -8(%rbp)
	jmp	.L187
.L174:
	movq	-16(%rbp), %rax
	movq	16(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, 16(%rax)
	movq	-16(%rbp), %rax
	movq	$0, 16(%rax)
	movq	$0, -24(%rbp)
	movl	$1, -44(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L187
.L162:
	movq	-24(%rbp), %rdx
	movq	-16(%rbp), %rax
	cmpq	%rax, %rdx
	jne	.L188
	movq	$32, -8(%rbp)
	jmp	.L187
.L188:
	movq	$34, -8(%rbp)
	jmp	.L187
.L176:
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L187
.L178:
	movl	-60(%rbp), %edx
	movq	-24(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	is_a_goal_node
	movl	%eax, -32(%rbp)
	movq	$21, -8(%rbp)
	jmp	.L187
.L184:
	cmpq	$0, -24(%rbp)
	je	.L190
	movq	$7, -8(%rbp)
	jmp	.L187
.L190:
	movq	$29, -8(%rbp)
	jmp	.L187
.L168:
	movq	-24(%rbp), %rax
	movl	4(%rax), %eax
	movl	%eax, -36(%rbp)
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -28(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L187
.L182:
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, %rdx
	movq	-16(%rbp), %rax
	cmpq	%rax, %rdx
	jne	.L192
	movq	$14, -8(%rbp)
	jmp	.L187
.L192:
	movq	$11, -8(%rbp)
	jmp	.L187
.L173:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L187
.L167:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L187
.L170:
	cmpl	$0, -32(%rbp)
	je	.L194
	movq	$6, -8(%rbp)
	jmp	.L187
.L194:
	movq	$12, -8(%rbp)
	jmp	.L187
.L157:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L196
	movq	$2, -8(%rbp)
	jmp	.L187
.L196:
	movq	$16, -8(%rbp)
	jmp	.L187
.L177:
	cmpl	$1, -44(%rbp)
	je	.L198
	movq	$20, -8(%rbp)
	jmp	.L187
.L198:
	movq	$34, -8(%rbp)
	jmp	.L187
.L175:
	movq	-24(%rbp), %rax
	movl	4(%rax), %eax
	movl	%eax, -36(%rbp)
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -28(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L187
.L161:
	movq	-16(%rbp), %rax
	movq	16(%rax), %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, (%rax)
	movq	$33, -8(%rbp)
	jmp	.L187
.L172:
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L187
.L180:
	movl	$1, -40(%rbp)
	movq	$25, -8(%rbp)
	jmp	.L187
.L159:
	cmpq	$0, -24(%rbp)
	je	.L200
	movq	$3, -8(%rbp)
	jmp	.L187
.L200:
	movq	$33, -8(%rbp)
	jmp	.L187
.L169:
	movq	$30, -8(%rbp)
	jmp	.L187
.L165:
	cmpl	$0, -40(%rbp)
	jne	.L202
	movq	$24, -8(%rbp)
	jmp	.L187
.L202:
	movq	$29, -8(%rbp)
	jmp	.L187
.L181:
	cmpq	$0, -24(%rbp)
	je	.L204
	movq	$8, -8(%rbp)
	jmp	.L187
.L204:
	movq	$28, -8(%rbp)
	jmp	.L187
.L160:
	movq	-16(%rbp), %rax
	jmp	.L210
.L179:
	movq	-24(%rbp), %rax
	movl	4(%rax), %eax
	cmpl	%eax, -36(%rbp)
	jle	.L207
	movq	$13, -8(%rbp)
	jmp	.L187
.L207:
	movq	$17, -8(%rbp)
	jmp	.L187
.L164:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$31, -8(%rbp)
	jmp	.L187
.L183:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, %edi
	call	exit@PLT
.L171:
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$34, -8(%rbp)
	jmp	.L187
.L211:
	nop
.L187:
	jmp	.L209
.L210:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE16:
	.size	remove_from_list, .-remove_from_list
	.globl	add_to_list
	.type	add_to_list, @function
add_to_list:
.LFB17:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$3, -8(%rbp)
.L229:
	cmpq	$9, -8(%rbp)
	ja	.L230
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L215(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L215(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L215:
	.long	.L230-.L215
	.long	.L222-.L215
	.long	.L221-.L215
	.long	.L220-.L215
	.long	.L230-.L215
	.long	.L219-.L215
	.long	.L218-.L215
	.long	.L231-.L215
	.long	.L216-.L215
	.long	.L214-.L215
	.text
.L216:
	movq	-32(%rbp), %rax
	movq	(%rax), %rdx
	movq	-16(%rbp), %rax
	movq	%rdx, 16(%rax)
	movq	$1, -8(%rbp)
	jmp	.L223
.L222:
	movq	-32(%rbp), %rax
	movq	(%rax), %rax
	movq	$0, 16(%rax)
	movq	$7, -8(%rbp)
	jmp	.L223
.L220:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L224
	movq	$6, -8(%rbp)
	jmp	.L223
.L224:
	movq	$5, -8(%rbp)
	jmp	.L223
.L214:
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L223
.L218:
	movq	-32(%rbp), %rax
	movq	(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, (%rax)
	movq	$1, -8(%rbp)
	jmp	.L223
.L219:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L223
.L221:
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	testq	%rax, %rax
	je	.L227
	movq	$9, -8(%rbp)
	jmp	.L223
.L227:
	movq	$8, -8(%rbp)
	jmp	.L223
.L230:
	nop
.L223:
	jmp	.L229
.L231:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE17:
	.size	add_to_list, .-add_to_list
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
